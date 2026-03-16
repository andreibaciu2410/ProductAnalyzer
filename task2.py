import hashlib
import os
from typing import List, Optional, Literal
from dotenv import load_dotenv
import instructor
import openai
from diskcache import Cache
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

load_dotenv()

cache = Cache(directory=os.getenv("CACHE_DIR", "./cache"))

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
MODEL = "llama-3.3-70b-versatile"


# =============================================================================
# MODELE
# =============================================================================

class ProductData(BaseModel):
    titlu: str
    descriere: str
    specificatii: str
    preț: str = ""
    extras_din: str


class FeatureComparison(BaseModel):
    feature_name: str
    produs_a_value: str
    produs_b_value: str
    rationale: str
    winner_score: int = Field(ge=1, le=10)
    winner: str = Field(pattern="^(A|B|Egal)$")
    relevant_pentru_user: bool


class Verdict(BaseModel):
    câștigător: str = Field(pattern="^(A|B|Egal)$")
    scor_a: int = Field(ge=0, le=100)
    scor_b: int = Field(ge=0, le=100)
    diferență_semificativă: bool
    argument_principal: str = Field(max_length=500)
    compromisuri: str = Field(max_length=500)


class ComparisonResult(BaseModel):
    produs_a_titlu: str
    produs_b_titlu: str
    features: List[FeatureComparison]
    verdict: Verdict
    preferinte_procesate: str


# Generatorul nu returnează CoT complet, ci un rezumat scurt al logicii.
class GeneratorOutput(BaseModel):
    gandire_rezumat: List[str] = Field(
        description="Pași scurți și non-sensibili ai raționamentului, max 6"
    )
    confidence: int = Field(ge=0, le=100)
    rezultat: ComparisonResult


class VerificationResult(BaseModel):
    valid: Literal["da", "nu", "nesigur"]
    motiv: str = Field(max_length=700)
    confidence_adecvat: bool
    probleme_identificate: List[str] = Field(default_factory=list)
    sugestie_corectie: str = Field(default="")


class ProductInput(BaseModel):
    sursa: str = Field(..., min_length=3)
    este_url: bool = False


class ComparisonRequest(BaseModel):
    produs_a: ProductInput
    produs_b: ProductInput
    preferinte: str = Field(..., min_length=5, max_length=1000)
    buget_maxim: Optional[int] = Field(None, ge=100)


# =============================================================================
# SCRAPING / PARSING
# =============================================================================

async def scrape_product(url: str) -> ProductData:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = await browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )

            await page.goto(url, wait_until="networkidle", timeout=25000)
            await page.wait_for_timeout(2000)

            html = await page.content()
            title = await page.title()
            await browser.close()

            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup.find_all([
                'script', 'style', 'nav', 'footer', 'header',
                'aside', 'noscript', 'iframe', 'svg', 'canvas',
                'button', 'input', 'form', 'select', 'textarea'
            ]):
                tag.decompose()

            content_parts = []

            h1 = soup.find('h1')
            if h1:
                product_title = h1.get_text(strip=True)
                if product_title:
                    content_parts.append(f"PRODUCT: {product_title}")

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_parts.append(f"DESCRIPTION: {meta_desc['content'][:500]}")

            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)

            for ul in soup.find_all(['ul', 'ol']):
                items = []
                for li in ul.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if len(item_text) > 5:
                        items.append(item_text)
                if items:
                    content_parts.append(" | ".join(items[:15]))

            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr')[:25]:
                    row_cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row_cells and any(cell for cell in row_cells):
                        rows.append(": ".join(row_cells[:2]))
                if rows:
                    content_parts.append("SPECS: " + " | ".join(rows[:10]))

            seen_fragments = set()
            final_content = []
            for part in content_parts:
                normalized = " ".join(part.lower().split())[:100]
                if normalized not in seen_fragments and len(part) > 20:
                    seen_fragments.add(normalized)
                    final_content.append(part)

            full_text = "\n\n".join(final_content[:40])

            return ProductData(
                titlu=title[:300] if title else url.split('/')[-1][:50],
                descriere=full_text[:6000],
                specificatii="",
                preț="",
                extras_din="beautifulsoup_clean"
            )

    except Exception as e:
        raise HTTPException(422, f"Scraping failed: {str(e)}")


def parse_text_input(text: str) -> ProductData:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ProductData(
        titlu=lines[0][:200] if lines else "Unknown",
        descriere="\n".join(lines[:20]),
        specificatii="",
        preț="",
        extras_din="text"
    )


# =============================================================================
# GENERATOR + VERIFICATOR
# =============================================================================

def build_comparison_payload(prod_a: ProductData, prod_b: ProductData, preferinte: str) -> str:
    return f"""
User preferințe: {preferinte}

PRODUS A: {prod_a.titlu}
Descriere: {prod_a.descriere[:6000]}
Spec: {prod_a.specificatii[:4000]}
Preț: {prod_a.preț}

PRODUS B: {prod_b.titlu}
Descriere: {prod_b.descriere[:6000]}
Spec: {prod_b.specificatii[:4000]}
Preț: {prod_b.preț}
""".strip()


async def genereaza_comparatie(
    prod_a: ProductData,
    prod_b: ProductData,
    preferinte: str,
    feedback_verificator: str = ""
) -> GeneratorOutput:
    system_prompt = """
Ești un expert în compararea produselor.

Generează:
1. un rezumat scurt al logicii (NU lanț de gândire complet; doar pași scurți),
2. un scor de confidence 0-100,
3. un ComparisonResult complet și valid.

Reguli:
- Compară strict pe baza datelor disponibile.
- Nu inventa specificații lipsă.
- Dacă informația e incompletă, reflectă asta în confidence și în verdict.
- features trebuie să includă doar aspecte relevante pentru preferințele userului.
"""

    user_prompt = f"""
{build_comparison_payload(prod_a, prod_b, preferinte)}

Feedback verificator anterior:
{feedback_verificator or "N/A"}

Returnează output structurat valid.
"""

    try:
        return instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=GeneratorOutput,
            max_retries=2,
            temperature=0,
            max_tokens=3500
        )
    except Exception as e:
        raise HTTPException(503, f"Generator error: {str(e)}")


async def verifica_comparatie(
    prod_a: ProductData,
    prod_b: ProductData,
    preferinte: str,
    generator_output: GeneratorOutput
) -> VerificationResult:
    system_prompt = """
Ești verificatorul logicii unui engine de comparație produse.

Primești:
- datele brute ale produselor,
- preferințele userului,
- rezultatul generatorului,
- confidence-ul generatorului.

Verifică:
1. dacă outputul respectă datele disponibile,
2. dacă verdictul urmează preferințele userului,
3. dacă confidence-ul este realist.

Răspunde cu:
- valid: da / nu / nesigur
- motiv
- confidence_adecvat
- probleme_identificate
- sugestie_corectie

Respinge dacă:
- există contradicții,
- câștigătorul nu e justificat,
- se afirmă specificații neconfirmate,
- confidence-ul e prea mare pentru date slabe.
"""

    user_prompt = f"""
DATE BRUTE:
{build_comparison_payload(prod_a, prod_b, preferinte)}

OUTPUT GENERATOR:
{generator_output.model_dump_json(indent=2, ensure_ascii=False)}
"""

    try:
        return instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=VerificationResult,
            max_retries=1,
            temperature=0,
            max_tokens=1800
        )
    except Exception as e:
        raise HTTPException(503, f"Verifier error: {str(e)}")


async def compara_cu_auto_verificare(
    prod_a: ProductData,
    prod_b: ProductData,
    preferinte: str,
    max_incercari: int = 3
) -> dict:
    feedback = ""
    istoric = []

    for incercare in range(1, max_incercari + 1):
        generated = await genereaza_comparatie(
            prod_a=prod_a,
            prod_b=prod_b,
            preferinte=preferinte,
            feedback_verificator=feedback
        )

        verificare = await verifica_comparatie(
            prod_a=prod_a,
            prod_b=prod_b,
            preferinte=preferinte,
            generator_output=generated
        )

        istoric.append({
            "incercare": incercare,
            "confidence_generator": generated.confidence,
            "validare": verificare.valid,
            "motiv": verificare.motiv
        })

        if verificare.valid == "da":
            return {
                "status": "validat",
                "incercari": incercare,
                "gandire_rezumat": generated.gandire_rezumat,
                "confidence": generated.confidence,
                "verification": verificare.model_dump(),
                "result": generated.rezultat,
                "istoric_validare": istoric
            }

        if verificare.valid == "nesigur" and generated.confidence <= 70:
            return {
                "status": "acceptat_cu_rezerve",
                "incercari": incercare,
                "gandire_rezumat": generated.gandire_rezumat,
                "confidence": generated.confidence,
                "verification": verificare.model_dump(),
                "result": generated.rezultat,
                "istoric_validare": istoric
            }

        feedback = f"""
Verificatorul a respins sau pus sub semnul întrebării outputul.

Motiv: {verificare.motiv}
Probleme: {'; '.join(verificare.probleme_identificate)}
Sugestie: {verificare.sugestie_corectie}

Refă rezultatul mai conservator și aliniază confidence-ul.
""".strip()

    raise HTTPException(
        422,
        detail={
            "message": "Nu s-a obținut un rezultat valid după maximul de încercări.",
            "istoric_validare": istoric
        }
    )


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(title="Product Comparison cu Auto-Verificare", version="4.0.0")


@app.post("/compare")
async def compare(request: ComparisonRequest):
    import time
    start = time.time()

    if request.produs_a.este_url:
        date_a = await scrape_product(request.produs_a.sursa)
    else:
        date_a = parse_text_input(request.produs_a.sursa)

    if request.produs_b.este_url:
        date_b = await scrape_product(request.produs_b.sursa)
    else:
        date_b = parse_text_input(request.produs_b.sursa)

    rezultat = await compara_cu_auto_verificare(
        prod_a=date_a,
        prod_b=date_b,
        preferinte=request.preferinte,
        max_incercari=3
    )

    rezultat["timp_ms"] = int((time.time() - start) * 1000)
    return rezultat


@app.get("/health")
async def health():
    try:
        client.models.list()
        ok = True
    except Exception:
        ok = False

    return {
        "status": "ok" if ok else "degraded",
        "model": MODEL,
        "pipeline": "generator + verifier + retry"
    }
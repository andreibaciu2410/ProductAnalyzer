78% of storage used … If you run out, you can't create, edit, and upload files. Get 100 GB of storage for RON 9.99 RON 2.49/month for 2 months (personalized price).
"""
Product Comparison Engine cu Instructor + OpenAI client pentru Ollama.
Garantează output structurat validat Pydantic prin Instructor.
"""

import hashlib
import os
from typing import List, Optional
from dotenv import load_dotenv
import instructor
import openai
from diskcache import Cache
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text
from pydantic import BaseModel, Field, field_validator

load_dotenv()
# =============================================================================
# CONFIGURARE
# =============================================================================

cache = Cache(directory=os.getenv("CACHE_DIR", "./cache"))

# Client OpenAI configurat pentru Ollama
# client = openai.OpenAI(
#     base_url=f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/v1",
#     api_key="ollama",  # Ollama ignoră, dar e necesar pentru client
# )

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", #f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/v1",
    api_key = os.getenv("GROQ_API_KEY"),  # Ollama ignoră, dar e necesar pentru client
)


# Patch cu Instructor pentru structured outputs
instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

MODEL = "llama-3.3-70b-versatile" #"qwen3:0.6b"


# =============================================================================
# MODELE PYDANTIC (Instructor le folosește pentru validare)
# =============================================================================

class ProductData(BaseModel):
    """Date extrase despre produs."""
    titlu: str = Field(description="Numele produsului")
    descriere: str = Field(description="Descriere scurtă")
    specificatii: str = Field(description="Specificații tehnice cheie")
    preț: str = Field(default="")
    extras_din: str = Field(description="'scraping' sau 'text'")


class FeatureComparison(BaseModel):
    """O linie din tabelul comparativ."""
    feature_name: str = Field(description="Numele caracteristicii")
    produs_a_value: str = Field(description="Valoare produs A")
    produs_b_value: str = Field(description="Valoare produs B")
    rationale: str = Field(description="Motivul pentru care s-a ales acest produs castigator la categoria respectiva")
    winner_score: int = Field(ge=1, le=10, description="Diferență 1-10")
    winner: str = Field(pattern="^(A|B|Egal)$")
    relevant_pentru_user: bool


class Verdict(BaseModel):
    """Verdict final al comparației."""
    câștigător: str = Field(pattern="^(A|B|Egal)$")
    scor_a: int = Field(ge=0, le=100, description="Scorul pentru primul produs")
    scor_b: int = Field(ge=0, le=100, description="Scorul pentru al doilea produs")
    diferență_semificativă: bool = Field(description="Daca exista o diferenta mare intre produse")
    argument_principal: str = Field(max_length=500)
    compromisuri: str = Field(max_length=500)


# AICI ESTE MAGIA INSTRUCTOR: response_model garantează structura
class ComparisonResult(BaseModel):
    """
    Model final pe care Instructor îl forțează din LLM.
    Dacă LLM returnează JSON invalid, Instructor retrimite automat.
    """
    produs_a_titlu: str = Field(description="Titlu produs A")
    produs_b_titlu: str = Field(description="Titlu produs B")
    features: List[FeatureComparison] = Field(description="Tabel comparativ")
    verdict: Verdict
    preferinte_procesate: str = Field(description="Rezumat preferințe user")


class ProductInput(BaseModel):
    sursa: str = Field(..., min_length=3)
    este_url: bool = Field(default=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sursa": "iPhone 15: A16, 6GB RAM, 48MP camera",
                "este_url": False
            }
        }


class ComparisonRequest(BaseModel):
    produs_a: ProductInput
    produs_b: ProductInput
    preferinte: str = Field(..., min_length=5, max_length=1000)
    buget_maxim: Optional[int] = Field(None, ge=100)


# =============================================================================
# SCRAPING
# =============================================================================

async def scrape_product(url: str) -> ProductData:
    """
    Scrapează orice pagină de produs cu BeautifulSoup.
    Elimină elemente inutile, păstrează tot conținutul relevant.
    """
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
            
            # BeautifulSoup pentru curățare
            soup = BeautifulSoup(html, 'html.parser')
            
            # ELIMINĂ elemente complet inutile
            for tag in soup.find_all([
                'script', 'style', 'nav', 'footer', 'header', 
                'aside', 'noscript', 'iframe', 'svg', 'canvas',
                'button', 'input', 'form', 'select', 'textarea',
                '[class*="cookie"]', '[class*="popup"]', '[class*="modal"]',
                '[id*="cookie"]', '[id*="popup"]', '[id*="modal"]',
                'advertisement', 'ad', 'banner'
            ]):
                tag.decompose()
            
            # EXTRAGE conținut util în ordinea importanței
            
            content_parts = []
            
            # 1. Titlu produs (din h1 sau meta)
            h1 = soup.find('h1')
            if h1:
                product_title = h1.get_text(strip=True)
                if product_title:
                    content_parts.append(f"PRODUCT: {product_title}")
            
            # 2. Descriere principală (meta description sau primele paragrafe)
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_parts.append(f"DESCRIPTION: {meta_desc['content'][:500]}")
            
            # 3. Toate paragrafele cu text substanțial
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 30:  # Ignoră paragrafe scurte (menu items, etc.)
                    content_parts.append(text)
            
            # 4. Liste (de obicei specs sau features)
            for ul in soup.find_all(['ul', 'ol']):
                items = []
                for li in ul.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if len(item_text) > 5:
                        items.append(item_text)
                if items:
                    content_parts.append(" | ".join(items[:15]))
            
            # 5. Tabele (specs tehnice)
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr')[:25]:
                    row_cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row_cells and any(cell for cell in row_cells):
                        rows.append(": ".join(row_cells[:2]))  # Label: Value
                if rows:
                    content_parts.append("SPECS: " + " | ".join(rows[:10]))
            
            # 6. Div-uri și secțiuni cu text dens (articole, descrieri)
            for element in soup.find_all(['div', 'section', 'article', 'main']):
                # Doar elemente cu mult text, nu containeri goi
                text = element.get_text(strip=True)
                if 200 < len(text) < 2000:  # Text substanțial dar nu enorm
                    # Verifică dacă nu e duplicat parțial
                    is_new = True
                    for existing in content_parts[-5:]:  # Compară cu ultimele 5
                        if text[:100] in existing or existing[:100] in text:
                            is_new = False
                            break
                    if is_new:
                        content_parts.append(text[:800])
            
            # Combină și deduplică
            seen_fragments = set()
            final_content = []
            
            for part in content_parts:
                # Normalizare pentru comparare
                normalized = " ".join(part.lower().split())[:100]
                if normalized not in seen_fragments and len(part) > 20:
                    seen_fragments.add(normalized)
                    final_content.append(part)
            
            # Text final curat
            full_text = "\n\n".join(final_content[:40])  # Limităm la 40 blocuri
            
            # Extrage preț simplu
            price = ""
            price_indicators = ['price', 'pret', 'preț', 'cost', '€', '$', 'lei', 'ron']
            for part in final_content[:10]:
                lower = part.lower()
                if any(ind in lower for ind in price_indicators):
                    # Caută pattern numeric lângă indicator
                    import re
                    matches = re.findall(r'[\d\s.,]+(?:lei|ron|€|\$|eur|usd)?', part, re.IGNORECASE)
                    if matches:
                        price = " ".join(matches[:2])
                        break
            
            return ProductData(
                titlu=title[:300] if title else url.split('/')[-1][:50],
                descriere=full_text[:6000],  # Tot conținutul curat
                specificatii="",  # Nu separăm - LLM extrage ce trebuie
                preț=price[:100],
                extras_din="beautifulsoup_clean"
            )
            
    except Exception as e:
        raise HTTPException(422, f"Scraping failed: {str(e)}")


def parse_text_input(text: str) -> ProductData:
    """Parsează input text liber."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ProductData(
        titlu=lines[0][:200] if lines else "Unknown",
        descriere='\n'.join(lines[:20]),
        specificatii="",
        preț="",
        extras_din="text"
    )


# =============================================================================
# INSTRUCTOR + LLM LOGIC
# =============================================================================

async def compară_produse_instructor(
    prod_a: ProductData,
    prod_b: ProductData,
    preferinte: str
) -> ComparisonResult:
    """
    Folosește Instructor pentru a forța output validat Pydantic.
    
    Instructor.wrap(client) + response_model=ComparisonResult
    = Garantat returnează obiect validat sau reîncearcă automat.
    """
    
    system_prompt = """Ești un expert în compararea produselor. 
                    Analizează datele reale ale produselor și compară-le STRICT pe criteriile userului.
                    Fii precis cu specificațiile tehnice extrase."""

    user_prompt = f"""Compară aceste produse pentru userul care vrea: "{preferinte}"

                    PRODUS A: {prod_a.titlu}
                    Descriere: {prod_a.descriere[:6000]}
                    Spec: {prod_a.specificatii[:4000]}

                    PRODUS B: {prod_b.titlu}
                    Descriere: {prod_b.descriere[:6000]}
                    Spec: {prod_b.specificatii[:4000]}

                    Generează tabel comparativ cu DOAR feature-urile relevante pentru preferințele userului.
                    Câștigătorul trebuie determinat bazat pe aceste preferințe, nu generic."""

    # INSTRUCTOR AICI: response_model=forțează structura exactă
    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=ComparisonResult,  # MAGIC: garantează validare Pydantic
            max_retries=2,  # Dacă e invalid, retrimite de 2 ori
            temperature=0,
            max_tokens=3000
        )
        return result
        
    except Exception as e:
        raise HTTPException(503, f"Instructor/LLM error: {str(e)}")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Product Comparison cu Instructor",
    description="""
    Comparare produs cu garantie Pydantic via Instructor.
    
    **Flow:**
    1. Extrage date (scraping sau text)
    2. Instructor + OpenAI client forțează output validat
    3. Returnează JSON garantat valid conform ComparisonResult
    
    **De ce Instructor?**
    - Garantează schema Pydantic sau reîncearcă automat
    - Nu mai e nevoie de parsing manual JSON
    - Tipuri Python native în tot codul
    """,
    version="3.0.0"
)


@app.post("/compare", response_model=ComparisonResult)
async def compare(request: ComparisonRequest):
    """
    Compară două produse cu Instructor garantat.
    
    **Exemple:**
    
    Cu URL:
    ```json
    {
        "produs_a": {"sursa": "https://example.com/laptop-a", "este_url": true},
        "produs_b": {"sursa": "https://example.com/laptop-b", "este_url": true},
        "preferinte": "Programare, 16GB RAM minim, tastatură bună, sub 2kg"
    }
    ```
    
    Cu text:
    ```json
    {
        "produs_a": {"sursa": "MacBook Air M3 8GB 256GB 1.24kg", "este_url": false},
        "produs_b": {"sursa": "ThinkPad X1 i7 16GB 512GB 1.13kg", "este_url": false},
        "preferinte": "Dezvoltare software și transport zilnic"
    }
    ```
    """
    import time
    start = time.time()
    
    # Cache key
    #cache_key = f"inv:{hashlib.sha256(request.model_dump_json().encode()).hexdigest()}"
    #cached = cache.get(cache_key)
    #if cached:
        # Reconstruim din cache
    #    return ComparisonResult.model_validate(cached)
    
    # Extrage date produse
    if request.produs_a.este_url:
        date_a = await scrape_product(request.produs_a.sursa)
    else:
        date_a = parse_text_input(request.produs_a.sursa)
        
    if request.produs_b.este_url:
        date_b = await scrape_product(request.produs_b.sursa)
    else:
        date_b = parse_text_input(request.produs_b.sursa)
    
    # INSTRUCTOR: Garantat returnează ComparisonResult validat
    result = await compară_produse_instructor(date_a, date_b, request.preferinte)
    
    # Adăugăm metadate
    result_dict = result.model_dump()
    result_dict["_timp_ms"] = int((time.time() - start) * 1000)
    result_dict["_din_cache"] = False
    
    # Salvăm în cache
    #cache.set(cache_key, result_dict, expire=3600*24)
    
    return result


@app.get("/health")
async def health():
    """Verificare stare."""
    try:
        # Test rapid Ollama
        client.models.list()
        ollama_ok = True
    except:
        ollama_ok = False
    
    return {
        "status": "ok" if ollama_ok else "degraded",
        "instructor": "active",
        "model": MODEL,
        "mode": "instructor-json"
    }


@app.delete("/cache")
async def clear_cache():
    """Golește cache."""
    cache.clear()
    return {"message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
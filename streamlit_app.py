# app.py
# -*- coding: utf-8 -*-
import asyncio
import json
import math
import re
import textwrap
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from openai import OpenAI

# ----------------------- USTAWIENIA STRONY ----------------------- #
st.set_page_config(page_title="Generator Metatagów SEO", page_icon="🏷️", layout="wide")

# ----------------------- AUTO-WYKRYWANIE HTTP/2 ------------------ #
try:
    import h2  # noqa: F401
    HTTP2_AVAILABLE = True
except Exception:
    HTTP2_AVAILABLE = False

# ----------------------- STAŁE / SŁOWNIKI ------------------------ #
CTA_WORDS = {
    "kup", "kupisz", "kupuj", "kup teraz", "sprawdź", "zobacz", "zamów",
    "kliknij", "odkryj", "poznaj", "przekonaj", "skorzystaj", "pobierz",
    "dodaj do koszyka", "zamawiaj", "porównaj", "zarezerwuj"
}

ATTRIBUTE_KEYS = {
    # normalizacja najczęściej spotykanych etykiet
    "materiał": ["materiał", "material", "surowiec"],
    "wiek": ["wiek", "od lat", "wiek dziecka", "age range", "wiek rekomendowany"],
    "format": ["format", "rozmiar", "wymiar", "wymiary", "size", "dimensions"],
    "liczba stron": ["liczba stron", "stron", "pages"],
    "oprawa": ["oprawa", "binding"],
    "kolekcja": ["kolekcja", "linia", "seria"],
    "kolor": ["kolor", "barwa", "color"],
    "pojemność": ["pojemność", "capacity", "objętość"],
    "typ": ["typ", "rodzaj", "type"],
}

LLM_MODEL = "gpt-5-nano"
MAX_TITLE = 60
MAX_DESC = 160

# ----------------------- DANE / MODELE --------------------------- #
@dataclass
class ProductData:
    url: str
    sku: str = ""
    title: str = ""
    isbn: str = ""
    category: str = ""
    attributes: Dict[str, str] = None
    description: str = ""
    error: Optional[str] = None

@dataclass
class MetaResult:
    url: str
    sku: str
    title: str
    isbn: str
    meta_title: str
    meta_description: str
    meta_title_length: int
    meta_desc_length: int
    error: Optional[str] = None

# ----------------------- POMOCNICZE ------------------------------ #
def to_host_brand(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        host = host.replace("www.", "")
        base = host.split(".")[0]
        return base.lower()
    except Exception:
        return ""

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def trim_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    words, out = text.split(), []
    for w in words:
        if len((" ".join(out + [w])).strip()) > limit:
            break
        out.append(w)
    out_text = " ".join(out).strip()
    return out_text if out_text else text[: max(0, limit - 1)].rstrip() + "…"

def normalize_title(title: str) -> str:
    t = title.replace("—", "-")
    t = t.replace("…", "")
    # usuń kropki w TITLE
    t = t.replace(".", "")
    t = re.sub(r"\s*-\s*", " - ", t)  # otoczenie myślnika spacjami
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def ensure_two_sentences(desc: str, attrib_hint: str = "") -> str:
    t = clean_text(desc)
    # usuń CTA
    for w in CTA_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()

    parts = re.split(r"(?<=[\.\?\!])\s+", t) if t else []
    parts = [p.strip() for p in parts if p.strip()]

    def make_fact_from_attr(hint: str) -> str:
        hint = clean_text(hint)
        hint = hint[:120]
        return hint or "Zawiera kluczowe cechy produktu"

    if len(parts) == 0:
        s1 = "Opis zawiera najważniejsze cechy produktu."
        s2 = make_fact_from_attr(attrib_hint) + "."
        t = f"{s1} {s2}"
    elif len(parts) == 1:
        s2 = make_fact_from_attr(attrib_hint) + "."
        s1 = parts[0]
        if not re.search(r"[\.!\?]$", s1):
            s1 += "."
        t = f"{s1} {s2}"
    else:
        s1, s2 = parts[0], parts[1]
        if not re.search(r"[\.!\?]$", s1):
            s1 += "."
        if not re.search(r"[\.!\?]$", s2):
            s2 += "."
        t = f"{s1} {s2}"

    return trim_words(t, MAX_DESC)

def compress_attributes(attrs: Dict[str, str]) -> str:
    if not attrs:
        return ""
    pairs = []
    for k, v in attrs.items():
        k = clean_text(k)
        v = clean_text(v)
        if k and v:
            pairs.append(f"{k}: {v}")
    return "; ".join(pairs)

# ----------------------- SCRAPING / PARSING ---------------------- #
def map_attribute_label(label: str) -> Optional[str]:
    label_lower = (label or "").strip().lower()
    for canonical, variants in ATTRIBUTE_KEYS.items():
        for v in variants:
            if label_lower == v or label_lower.startswith(v):
                return canonical
    return None

def extract_jsonld_product(soup: bs) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
            if isinstance(data, dict):
                data = [data]
            for d in data:
                t = d.get("@type")
                if isinstance(t, list):
                    t = next((x for x in t if isinstance(x, str)), None)
                if t in ("Product", "Book"):
                    items.append(d)
        except Exception:
            continue
    return items[0] if items else {}

def extract_isbn(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b97[89][- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?\d\b", text)
    return m.group(0).replace(" ", "").replace("-", "") if m else ""

def parse_attributes_from_details(details_root: Tag) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    if not details_root:
        return attrs
    for li in details_root.find_all("li"):
        txt = clean_text(li.get_text(" ", strip=True))
        if not txt:
            continue
        if ":" in txt:
            lab, val = txt.split(":", 1)
        elif "–" in txt:
            lab, val = txt.split("–", 1)
        elif "-" in txt:
            lab, val = txt.split("-", 1)
        else:
            continue
        lab = clean_text(lab)
        val = clean_text(val)
        key = map_attribute_label(lab)
        if key and val:
            attrs.setdefault(key, val)
    return attrs

async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=httpx.Timeout(20.0, connect=5.0))
    r.raise_for_status()
    return r.text

async def scrape_product(url: str) -> ProductData:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MetaTagsBot/1.0; +https://example.com/bot)",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    }
    try:
        # AUTO: http2=HTTP2_AVAILABLE – brak h2 => przełącza na HTTP/1.1
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, http2=HTTP2_AVAILABLE) as client:
            html = await fetch_html(client, url)
    except Exception as e:
        return ProductData(url=url, error=f"Błąd pobierania: {e}")

    try:
        soup = bs(html, "lxml")
    except Exception:
        soup = bs(html, "html.parser")

    # Tytuł
    title_tag = soup.find("h1") or soup.find("h1", attrs={"itemprop": "name"})
    title = clean_text(title_tag.get_text(strip=True)) if title_tag else ""

    # JSON-LD
    ld = extract_jsonld_product(soup)
    ld_name = clean_text(ld.get("name", "")) if ld else ""
    ld_desc = clean_text(ld.get("description", "")) if ld else ""
    ld_category = clean_text(ld.get("category", "")) if ld else ""
    ld_isbn = clean_text(ld.get("isbn", "")) if ld else ""
    if not ld_isbn:
        try:
            ap = ld.get("additionalProperty") or []
            if isinstance(ap, dict):
                ap = [ap]
            for p in ap:
                if str(p.get("name", "")).lower() == "isbn":
                    ld_isbn = clean_text(p.get("value", ""))
                    break
        except Exception:
            pass

    # Opis
    description_text = ""
    desc_candidates = [
        ("div", {"class": "desc-container"}),
        ("div", {"id": "product-description"}),
        ("div", {"itemprop": "description"}),
        ("section", {"id": "description"}),
        ("div", {"class": "product-description"}),
        ("article", {}),
    ]
    for tag, attrs in desc_candidates:
        node = soup.find(tag, attrs=attrs)
        if node:
            art = node.find("article") or node
            description_text = clean_text(art.get_text(separator="\n", strip=True))
            if description_text:
                break

    # Smyk spec
    if "smyk.com" in url and not description_text:
        smyk_desc_div = soup.find("div", attrs={"data-testid": "box-attributes__simple"})
        if smyk_desc_div:
            for p_tag in smyk_desc_div.find_all("p"):
                if p_tag.find("span", string=lambda x: x and "Nr produktu:" in x):
                    p_tag.decompose()
            description_text = clean_text(smyk_desc_div.get_text(separator="\n", strip=True))

    # Szczegóły / atrybuty
    details_root = (
        soup.find("div", id="szczegoly")
        or soup.find("div", class_="product-features")
        or soup.find("ul", class_="bullet")
        or soup.find("div", class_="box-attributes__not-simple")
    )
    attributes = parse_attributes_from_details(details_root) if details_root else {}

    # ISBN
    isbn = ld_isbn or extract_isbn(ld_desc or description_text or html)

    # Kategoria
    category = ld_category
    if not category:
        bc = soup.find("nav", {"aria-label": re.compile("breadcrumb", re.I)}) or soup.find("ul", class_="breadcrumbs")
        if bc:
            cat = clean_text(bc.get_text(" > ", strip=True))
            category = cat.split(">")[-1].strip() if ">" in cat else cat

    # Tytuł/Opis – preferuj JSON-LD
    if ld_name and len(ld_name) > 4:
        title = ld_name
    description = ld_desc if len(ld_desc) > 30 else description_text

    return ProductData(
        url=url,
        title=title,
        isbn=isbn,
        category=category,
        attributes=attributes,
        description=description,
        error=None,
    )

# ----------------------- LLM / PROMPT ---------------------------- #
def build_system_prompt() -> str:
    return textwrap.dedent("""
    Jesteś ekspertem SEO. Tworzysz metatagi e-commerce po polsku.

    WYMAGANIA META TITLE:
    - ≤ 60 znaków (spacje wliczone)
    - 1 fraza kluczowa na start + 1–2 cechy produktu
    - Zwykły myślnik "-" wyłącznie, bez kropek i brandu/sklepu
    - Bez CTA

    WYMAGANIA META DESCRIPTION:
    - ≤ 160 znaków
    - Dokładnie 2 krótkie zdania, wyłącznie informacyjne
    - Bez CTA i bez nazwy sklepu/brandu
    - Naturalne słowa kluczowe

    ZWRÓĆ WYŁĄCZNIE JSON:
    {"meta_title":"...","meta_description":"..."}
    """).strip()

def build_user_prompt(pd: ProductData, brand_block: str) -> str:
    attrs_str = compress_attributes(pd.attributes or {})
    desc_snippet = clean_text(pd.description)[:800]
    return textwrap.dedent(f"""
    DANE PRODUKTU (oczyszczone):
    Tytuł: {pd.title or "brak"}
    Kategoria: {pd.category or "brak"}
    Atrybuty kluczowe: {attrs_str or "brak"}
    ISBN: {pd.isbn or "brak"}
    Opis (skrót): {desc_snippet or "brak"}

    Nazwy zakazane (nie mogą się pojawić): {brand_block or "brak"}
    Stwórz metatagi wg wymagań.
    """).strip()

def postprocess_llm_output(meta_title: str, meta_description: str, banned_words: List[str]) -> Tuple[str, str]:
    meta_title = normalize_title(meta_title)
    for w in banned_words:
        if not w:
            continue
        meta_title = re.sub(rf"\b{re.escape(w)}\b", "", meta_title, flags=re.I)
        meta_description = re.sub(rf"\b{re.escape(w)}\b", "", meta_description, flags=re.I)
    meta_description = ensure_two_sentences(meta_description, "")
    meta_title = trim_words(meta_title, MAX_TITLE)
    meta_description = trim_words(meta_description, MAX_DESC)
    meta_title = re.sub(r"\s{2,}", " ", meta_title).strip(" -")
    meta_description = re.sub(r"\s{2,}", " ", meta_description).strip()
    return meta_title, meta_description

async def generate_for_product(client: OpenAI, pd: ProductData, semaphore: asyncio.Semaphore) -> MetaResult:
    if pd.error:
        return MetaResult(
            url=pd.url, sku="", title=pd.title, isbn=pd.isbn,
            meta_title="", meta_description="", meta_title_length=0, meta_desc_length=0,
            error=pd.error
        )
    system_prompt = build_system_prompt()
    brand = to_host_brand(pd.url)
    user_prompt = build_user_prompt(pd, brand)

    try:
        async with semaphore:
            def call_openai():
                return client.responses.create(
                    model=LLM_MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    seed=42,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "low"},
                )
            resp = await asyncio.to_thread(call_openai)

        txt = resp.output_text if hasattr(resp, "output_text") else ""
        data = {}
        try:
            data = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.S)
            if m:
                data = json.loads(m.group(0))
            else:
                raise ValueError(f"Nie udało się sparsować JSON: {txt[:200]}...")

        mt = clean_text(data.get("meta_title", ""))
        md = clean_text(data.get("meta_description", ""))

        banned = [brand]
        mt, md = postprocess_llm_output(mt, md, banned)

        return MetaResult(
            url=pd.url, sku=pd.sku or "", title=pd.title or "", isbn=pd.isbn or "",
            meta_title=mt, meta_description=md,
            meta_title_length=len(mt), meta_desc_length=len(md),
            error=None
        )
    except Exception as e:
        return MetaResult(
            url=pd.url, sku=pd.sku or "", title=pd.title or "", isbn=pd.isbn or "",
            meta_title="", meta_description="", meta_title_length=0, meta_desc_length=0,
            error=f"Błąd LLM: {e}"
        )

# ----------------------- PIPELINE ------------------------------- #
async def run_pipeline(urls: List[str], skus: List[str], llm_client: OpenAI) -> List[MetaResult]:
    scrape_tasks = [scrape_product(u) for u in urls]
    scraped: List[ProductData] = await asyncio.gather(*scrape_tasks)

    for i, pd_obj in enumerate(scraped):
        pd_obj.sku = skus[i] if i < len(skus) else ""

    sem = asyncio.Semaphore(3)  # 3 równoległe wywołania LLM
    gen_tasks = [generate_for_product(llm_client, pd_obj, sem) for pd_obj in scraped]
    results: List[MetaResult] = []
    completed = 0
    total = len(gen_tasks)

    for coro in asyncio.as_completed(gen_tasks):
        r = await coro
        results.append(r)
        completed += 1
        st.session_state.progress_placeholder.progress(
            completed / total,
            text=f"Generowanie metatagów: {completed}/{total}"
        )

    idx_map = {u: i for i, u in enumerate(urls)}
    results_sorted = sorted(results, key=lambda x: idx_map.get(x.url, 10**9))
    return results_sorted

# ----------------------- UI ------------------------------------ #
st.title("🏷️ Generator Metatagów SEO – Tryb Wsadowy (wersja PRO)")
st.markdown("Wygeneruj **meta title** i **meta description** na bazie danych konkurencji – stabilny JSON, walidacja i semantyka.")

st.sidebar.header("📊 Limity SEO")
st.sidebar.metric("Meta Title", f"max {MAX_TITLE} znaków")
st.sidebar.metric("Meta Description", f"max {MAX_DESC} znaków")
st.sidebar.markdown("---")
st.sidebar.subheader("✅ Standardy SEO")
st.sidebar.markdown("""
- **Meta Title:** zwykły myślnik "-", brak kropek i brandu
- **Meta Description:** dokładnie 2 zdania, bez CTA, obiektywne fakty
- Naturalne słowa kluczowe, skupienie na produkcie
""")

st.info("📝 Wklej linki do produktów i (opcjonalnie) kody SKU – jeden na linię.")

col1, col2 = st.columns([2, 1])
with col1:
    urls_input = st.text_area(
        "🔗 Linki do produktów (jeden na linię)",
        height=260,
        placeholder="https://example.com/produkt-1\nhttps://example.com/produkt-2",
        key="urls",
    )
with col2:
    skus_input = st.text_area(
        "🏷️ Kody SKU (opcjonalne, jeden na linię)",
        height=260,
        placeholder="SKU-001\nSKU-002",
        key="skus",
        help="Opcjonalne. Dopasowywane w kolejności do linków.",
    )

if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ Brak klucza API OpenAI w secrets. Skonfiguruj `OPENAI_API_KEY`.")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results: List[MetaResult] = []

client = OpenAI()

col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    gen_clicked = st.button("🚀 Generuj metatagi", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Wyczyść", use_container_width=True):
        st.session_state.results = []
        st.rerun()

# Komunikat o trybie HTTP/2 (informacyjnie)
if HTTP2_AVAILABLE:
    st.caption("🔌 HTTP/2: włączony (wykryto pakiet `h2`).")
else:
    st.caption("🔌 HTTP/2: wyłączony (brak pakietu `h2`; działa HTTP/1.1).")

st.session_state.progress_placeholder = st.empty()

if gen_clicked:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    skus = [s.strip() for s in skus_input.splitlines() if s.strip()]

    if not urls:
        st.error("❌ Podaj przynajmniej jeden link do produktu!")
    else:
        if len(skus) < len(urls):
            skus.extend([""] * (len(urls) - len(skus)))
        elif len(skus) > len(urls):
            st.warning(f"⚠️ SKU ({len(skus)}) > linków ({len(urls)}). Nadmiarowe SKU zostaną zignorowane.")
            skus = skus[: len(urls)]

        st.session_state.progress_placeholder.progress(0.0, text="Scraping danych produktów...")
        try:
            results = asyncio.run(run_pipeline(urls, skus, client))
            st.session_state.results = results
            st.session_state.progress_placeholder.progress(1.0, text="✅ Zakończono generowanie!")
            st.success(f"Wygenerowano metatagi dla {len(results)} produktów.")
        except Exception as e:
            st.error(f"❌ Błąd przetwarzania: {e}")

# ----------------------- WYNIKI / STATYSTYKI -------------------- #
results = st.session_state.results
if results:
    st.markdown("---")
    st.header("📊 Wyniki")

    successful = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔗 Wszystkie", len(results))
    c2.metric("✅ Sukces", len(successful))
    c3.metric("❌ Błędy", len(errors))
    if successful:
        avg_title = sum(r.meta_title_length for r in successful) / len(successful)
        avg_desc = sum(r.meta_desc_length for r in successful) / len(successful)
        c4.metric("📏 Śr. długość title", f"{avg_title:.0f} zn.")

    if successful:
        df = pd.DataFrame([
            {
                "URL": r.url,
                "SKU": r.sku,
                "ISBN": r.isbn,
                "Tytuł produktu": r.title,
                "Meta Title": r.meta_title,
                "Meta Description": r.meta_description,
                "Długość Title": r.meta_title_length,
                "Długość Description": r.meta_desc_length,
            }
            for r in successful
        ])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Pobierz wyniki CSV",
            data=csv,
            file_name="metatagi_seo.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📋 Tabela wyników")

    show_filter = st.radio(
        "Pokaż:",
        ["Wszystkie", "Tylko sukces", "Tylko błędy"],
        horizontal=True,
    )

    if show_filter == "Tylko sukces":
        displayed = successful
    elif show_filter == "Tylko błędy":
        displayed = errors
    else:
        displayed = results

    table_rows: List[Dict[str, Any]] = []
    for r in displayed:
        if r.error:
            table_rows.append({
                "Status": "❌",
                "SKU": r.sku or "-",
                "ISBN": r.isbn or "-",
                "Meta Title": f"BŁĄD: {r.error[:80]}...",
                "Meta Description": "-",
                "Długość T": "-",
                "Długość D": "-",
            })
        else:
            t_status = "🟢" if r.meta_title_length <= MAX_TITLE else "🟡"
            d_status = "🟢" if r.meta_desc_length <= MAX_DESC else "🟡"
            table_rows.append({
                "Status": f"{t_status}{d_status}",
                "SKU": r.sku or "-",
                "ISBN": r.isbn or "-",
                "Meta Title": r.meta_title,
                "Meta Description": r.meta_description,
                "Długość T": f"{r.meta_title_length}/{MAX_TITLE}",
                "Długość D": f"{r.meta_desc_length}/{MAX_DESC}",
            })

    if table_rows:
        df_display = pd.DataFrame(table_rows)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("", width="small"),
                "SKU": st.column_config.TextColumn("SKU", width="small"),
                "ISBN": st.column_config.TextColumn("ISBN", width="small"),
                "Meta Title": st.column_config.TextColumn("Meta Title", width="large"),
                "Meta Description": st.column_config.TextColumn("Meta Description", width="large"),
                "Długość T": st.column_config.TextColumn("Dł. T", width="small"),
                "Długość D": st.column_config.TextColumn("Dł. D", width="small"),
            },
        )

    with st.expander("🛠️ Diagnostyka (dla ciekawych)"):
        st.write("Poniżej surowe dane wejściowe po scrapingu (pierwsze 3 pozycje):")
        diag = []
        for r in results[:3]:
            diag.append(asdict(r))
        st.json(diag)

# ----------------------- STOPKA ------------------------------- #
st.markdown("---")
st.markdown("🔧 **Generator Metatagów SEO – wersja PRO** | JSON-LD, walidacja 2 zdań, anty-CTA, myślnik „-”, bez brandu | Powered by OpenAI GPT-5-nano")

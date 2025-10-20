import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ------------- USTAWIENIA STRONY ------------- #
st.set_page_config(page_title="Generator Metatagów SEO", page_icon="🏷️", layout="wide")

# ------------- POBIERANIE DANYCH ------------- #
def get_product_data(url):
    """Scrapuje dane produktu ze strony."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = bs(response.text, 'html.parser')

        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''

        # Pobieranie ISBN z sekcji szczegółów
        isbn = ""
        details_div = soup.find("div", id="accordion-content__szczegóły")
        if details_div:
            ul = details_div.find("ul", class_="bullet")
            if ul:
                for li in ul.find_all("li", class_="display-detail"):
                    li_text = li.get_text(strip=True)
                    if li_text.startswith("ISBN:"):
                        strong_tag = li.find("strong")
                        if strong_tag:
                            isbn = strong_tag.get_text(strip=True)
                        break

        details_text = ""
        description_text = ""

        # Logika scrapowania dla smyk.com
        if 'smyk.com' in url:
            smyk_desc_div = soup.find("div", attrs={"data-testid": "box-attributes__simple"})
            if smyk_desc_div:
                for p_tag in smyk_desc_div.find_all("p"):
                    if p_tag.find("span", string=lambda x: x and "Nr produktu:" in x):
                        p_tag.decompose()
                description_text = smyk_desc_div.get_text(separator="\n", strip=True)

            smyk_attributes_div = soup.find("div", class_="box-attributes__not-simple")
            if smyk_attributes_div:
                attributes_list = []
                items = smyk_attributes_div.find_all("div", class_="box_attributes__spec-item")
                for item in items:
                    label_tag = item.find("span", class_="box-attributes-list__label--L")
                    value_tag = item.find("span", class_="box-attributes-list__atribute--L")
                    if label_tag and value_tag:
                        label = label_tag.get_text(strip=True)
                        value = value_tag.get_text(strip=True)
                        if label and value:
                            attributes_list.append(f"{label}: {value}")
                
                if attributes_list:
                    details_text = "\n".join(attributes_list)
        
        # Uniwersalne scrapowanie dla innych stron
        if not description_text:
            details_div = soup.find("div", id="szczegoly") or soup.find("div", class_="product-features")
            if details_div:
                ul = details_div.find("ul", class_="bullet") or details_div.find("ul")
                if ul:
                    li_elements = ul.find_all("li")
                    details_list = [li.get_text(separator=" ", strip=True) for li in li_elements]
                    details_text = "\n".join(details_list)
            
            description_div = soup.find("div", class_="desc-container")
            if description_div:
                article = description_div.find("article")
                if article:
                    nested_article = article.find("article")
                    if nested_article:
                        description_text = nested_article.get_text(separator="\n", strip=True)
                    else:
                        description_text = article.get_text(separator="\n", strip=True)
                else:
                    description_text = description_div.get_text(separator="\n", strip=True)

        if not description_text:
            alt_desc_div = soup.find("div", id="product-description")
            if alt_desc_div:
                description_text = alt_desc_div.get_text(separator="\n", strip=True)

        description_text = " ".join(description_text.split())

        if not description_text and not details_text:
            return {
                'title': title,
                'isbn': isbn,
                'details': '',
                'description': '',
                'error': "Nie udało się pobrać danych produktu."
            }
        
        return {
            'title': title,
            'isbn': isbn,
            'details': details_text,
            'description': description_text,
            'error': None
        }
    except Exception as e:
        return {
            'title': '',
            'isbn': '',
            'details': '',
            'description': '',
            'error': f"Błąd pobierania: {str(e)}"
        }

# ------------- GENEROWANIE METATAGÓW ------------- #
def generate_meta_tags(product_data, client):
    """Generuje meta title i meta description."""
    try:
        title = product_data.get('title', '')
        details = product_data.get('details', '')
        description = product_data.get('description', '')
        
        system_prompt = """Jesteś ekspertem SEO specjalizującym się w optymalizacji metatagów dla e-commerce.

ZASADY TWORZENIA META TITLE:
- Maksymalnie 60 znaków (włącznie ze spacjami)
- Zacznij od najważniejszego słowa kluczowego
- Bądź konkretny i zachęcający do kliknięcia
- NIGDY nie dodawaj na końcu nazwy sklepu, brandu ani domeny
- NIE używaj separatorów z nazwą sklepu na końcu
- Skup się tylko na produkcie i jego kluczowych cechach

ZASADY TWORZENIA META DESCRIPTION:
- Maksymalnie 160 znaków (włącznie ze spacjami)
- DOKŁADNIE DWA krótkie zdania informacyjne
- Pierwsze zdanie: główna cecha/wartość produktu
- Drugie zdanie: dodatkowa korzyść lub call-to-action
- Naturalne użycie słów kluczowych
- Jasno komunikuj wartość produktu
- Bez nazwy sklepu ani brandu

Pisz WYŁĄCZNIE po polsku. Odpowiedź musi być w formacie:
Meta title: [treść]
Meta description: [treść]"""

        user_prompt = f"""Produkt: {title}

Szczegóły: {details[:500] if details else 'brak'}

Opis: {description[:800] if description else 'brak'}

Stwórz zoptymalizowane SEO metatagi dla tego produktu."""

        full_input = f"{system_prompt}\n\n{user_prompt}"

        response = client.responses.create(
            model="gpt-5-nano",
            input=full_input,
            reasoning={"effort": "medium"},
            text={"verbosity": "low"}
        )
        
        result = response.output_text
        meta_title = ""
        meta_description = ""
        
        for line in result.splitlines():
            line = line.strip()
            if line.lower().startswith("meta title:"):
                meta_title = line[len("meta title:"):].strip()
            elif line.lower().startswith("meta description:"):
                meta_description = line[len("meta description:"):].strip()
        
        # Walidacja długości
        if len(meta_title) > 60:
            meta_title = meta_title[:57] + "..."
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        return meta_title, meta_description
    except Exception as e:
        return f"BŁĄD: {str(e)}", f"BŁĄD: {str(e)}"

# ------------- PRZETWARZANIE RÓWNOLEGŁE ------------- #
def process_single_product(url, sku, client):
    """Przetwarza jeden produkt: scrapuje dane i generuje metatagi."""
    try:
        product_data = get_product_data(url)
        
        if product_data['error']:
            return {
                'url': url,
                'sku': sku,
                'title': product_data.get('title', ''),
                'isbn': product_data.get('isbn', ''),
                'meta_title': '',
                'meta_description': '',
                'error': product_data['error']
            }
        
        meta_title, meta_description = generate_meta_tags(product_data, client)
        
        if "BŁĄD:" in meta_title:
            return {
                'url': url,
                'sku': sku,
                'title': product_data.get('title', ''),
                'isbn': product_data.get('isbn', ''),
                'meta_title': '',
                'meta_description': '',
                'error': meta_title
            }
        
        return {
            'url': url,
            'sku': sku,
            'title': product_data.get('title', ''),
            'isbn': product_data.get('isbn', ''),
            'meta_title': meta_title,
            'meta_description': meta_description,
            'meta_title_length': len(meta_title),
            'meta_desc_length': len(meta_description),
            'error': None
        }
    except Exception as e:
        return {
            'url': url,
            'sku': sku,
            'title': '',
            'isbn': '',
            'meta_title': '',
            'meta_description': '',
            'error': f"Nieoczekiwany błąd: {str(e)}"
        }

# ------------- INICJALIZACJA ------------- #
if 'results' not in st.session_state:
    st.session_state.results = []

if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ Brak klucza API OpenAI w secrets. Skonfiguruj OPENAI_API_KEY.")
    st.stop()

client = OpenAI()

# ------------- INTERFEJS UŻYTKOWNIKA ------------- #
st.title('🏷️ Generator Metatagów SEO - Tryb Wsadowy')
st.markdown("Wygeneruj zoptymalizowane meta title i meta description dla wielu produktów jednocześnie.")

# Sidebar z informacjami
st.sidebar.header("📊 Limity SEO")
st.sidebar.metric("Meta Title", "max 60 znaków")
st.sidebar.metric("Meta Description", "max 160 znaków")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Wskazówka:** Zielony status 🟢 oznacza poprawną długość, żółty 🟡 przekroczenie limitu.")
st.sidebar.markdown("---")
st.sidebar.subheader("✅ Standardy SEO")
st.sidebar.markdown("""
- **Meta Title:** Bez nazwy sklepu/brandu
- **Meta Description:** Dokładnie 2 zdania
- Skupienie na produkcie i jego wartości
""")

# Główna zawartość
st.info("📝 Wklej linki do produktów i wygeneruj dla nich zoptymalizowane metatagi SEO.")

col1, col2 = st.columns([2, 1])

with col1:
    urls_input = st.text_area(
        "🔗 Linki do produktów (jeden na linię)",
        height=300,
        placeholder="https://example.com/produkt-1\nhttps://example.com/produkt-2\nhttps://example.com/produkt-3",
        key="urls"
    )

with col2:
    skus_input = st.text_area(
        "🏷️ Kody SKU (opcjonalne, jeden na linię)",
        height=300,
        placeholder="SKU-001\nSKU-002\nSKU-003",
        key="skus",
        help="Opcjonalne pole identyfikacyjne produktu"
    )

col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    if st.button("🚀 Generuj metatagi", type="primary", use_container_width=True):
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        skus = [sku.strip() for sku in skus_input.splitlines() if sku.strip()]
        
        if not urls:
            st.error("❌ Podaj przynajmniej jeden link do produktu!")
        else:
            # Jeśli SKU nie podano, uzupełnij pustymi stringami
            if len(skus) < len(urls):
                skus.extend([''] * (len(urls) - len(skus)))
            elif len(skus) > len(urls):
                st.warning(f"⚠️ Liczba SKU ({len(skus)}) jest większa niż linków ({len(urls)}). Ignoruję nadmiarowe SKU.")
                skus = skus[:len(urls)]
            
            st.session_state.results = []
            
            progress_bar = st.progress(0, text="Rozpoczynam generowanie metatagów...")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_data = {
                    executor.submit(process_single_product, url, sku, client): (url, sku)
                    for url, sku in zip(urls, skus)
                }
                
                results_temp = []
                for i, future in enumerate(as_completed(future_to_data)):
                    result = future.result()
                    results_temp.append(result)
                    progress_bar.progress(
                        (i + 1) / len(future_to_data),
                        text=f"Przetworzono {i+1}/{len(future_to_data)}"
                    )
            
            # Sortuj według kolejności oryginalnych URL-i
            st.session_state.results = sorted(results_temp, key=lambda x: urls.index(x['url']))
            progress_bar.progress(1.0, text="✅ Zakończono!")
            st.success(f"Wygenerowano metatagi dla {len(st.session_state.results)} produktów!")

with col_btn2:
    if st.button("🗑️ Wyczyść", use_container_width=True):
        st.session_state.results = []
        st.rerun()

# Wyświetlanie wyników
if st.session_state.results:
    st.markdown("---")
    st.header("📊 Wyniki")
    
    results = st.session_state.results
    successful = [r for r in results if r['error'] is None]
    errors = [r for r in results if r['error'] is not None]
    
    # Statystyki
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔗 Wszystkie", len(results))
    col2.metric("✅ Sukces", len(successful))
    col3.metric("❌ Błędy", len(errors))
    
    if successful:
        avg_title_len = sum(r['meta_title_length'] for r in successful) / len(successful)
        avg_desc_len = sum(r['meta_desc_length'] for r in successful) / len(successful)
        col4.metric("📏 Śr. długość title", f"{avg_title_len:.0f} zn.")
    
    # Eksport do CSV
    if successful:
        df = pd.DataFrame([
            {
                'URL': r['url'],
                'SKU': r['sku'],
                'ISBN': r['isbn'],
                'Tytuł produktu': r['title'],
                'Meta Title': r['meta_title'],
                'Meta Description': r['meta_description'],
                'Długość Title': r['meta_title_length'],
                'Długość Description': r['meta_desc_length']
            }
            for r in successful
        ])
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Pobierz wyniki jako CSV",
            csv,
            "metatagi_seo.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Szczegółowe wyniki
    st.markdown("---")
    st.subheader("📋 Szczegółowe wyniki")
    
    # Filtrowanie
    show_filter = st.radio(
        "Pokaż:",
        ["Wszystkie", "Tylko sukces", "Tylko błędy"],
        horizontal=True
    )
    
    if show_filter == "Tylko sukces":
        displayed_results = successful
    elif show_filter == "Tylko błędy":
        displayed_results = errors
    else:
        displayed_results = results
    
    # Wyświetlanie kart z wynikami
    for i, result in enumerate(displayed_results, 1):
        if result['error']:
            with st.expander(f"❌ [{i}] Błąd: {result['url'][:60]}...", expanded=False):
                st.error(result['error'])
                st.write(f"**URL:** {result['url']}")
                if result['sku']:
                    st.write(f"**SKU:** {result['sku']}")
                if result['isbn']:
                    st.write(f"**ISBN:** {result['isbn']}")
        else:
            title_status = "🟢" if result['meta_title_length'] <= 60 else "🟡"
            desc_status = "🟢" if result['meta_desc_length'] <= 160 else "🟡"
            
            with st.expander(
                f"✅ [{i}] {result['title'][:50]}... {title_status} {desc_status}",
                expanded=False
            ):
                st.write(f"**🌐 URL:** {result['url']}")
                if result['sku']:
                    st.write(f"**🏷️ SKU:** {result['sku']}")
                if result['isbn']:
                    st.write(f"**📚 ISBN:** {result['isbn']}")
                st.write(f"**📦 Tytuł produktu:** {result['title']}")
                
                st.markdown("---")
                
                # Meta Title
                col_t1, col_t2 = st.columns([4, 1])
                with col_t1:
                    st.markdown(f"**Meta Title:**")
                    if result['meta_title_length'] <= 60:
                        st.success(result['meta_title'])
                    else:
                        st.warning(result['meta_title'])
                with col_t2:
                    st.metric("Długość", f"{result['meta_title_length']}/60")
                
                # Meta Description
                col_d1, col_d2 = st.columns([4, 1])
                with col_d1:
                    st.markdown(f"**Meta Description:**")
                    if result['meta_desc_length'] <= 160:
                        st.success(result['meta_description'])
                    else:
                        st.warning(result['meta_description'])
                with col_d2:
                    st.metric("Długość", f"{result['meta_desc_length']}/160")

# Stopka
st.markdown("---")
st.markdown("🔧 **Generator Metatagów SEO** | Powered by OpenAI GPT-5-nano")

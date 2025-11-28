import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# --- НАЛАШТУВАННЯ ---

BASE_URL = "https://www.yachtcharterfleet.com"
START_URL = "https://www.yachtcharterfleet.com/charter/superyachts-for-charter"
PAGES_TO_SCRAPE_PER_TYPE = 42
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

YACHT_TYPES = {
    1: 'Motor Yachts', 2: 'Sailing Yachts', 3: 'Expedition Yachts',
    4: 'Classic Yachts', 5: 'Open Yachts', 6: 'Catamarans',
    7: 'Sport Fishing', 8: 'Gulet Yachts'
}

# --- ФУНКЦІЯ ЗБОРУ ПОСИЛАНЬ (залишається без змін, вона працює коректно) ---
def get_yacht_links(list_url):
    links = []
    try:
        response = requests.get(list_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results_container = soup.find('div', id='yacht-results-listing')
        if results_container:
            yacht_cards = results_container.find_all('div', class_='jsYachtSearchResult')
            for card in yacht_cards:
                link_tag = card.find('a', class_='searchImageLink', href=True)
                if link_tag:
                    links.append(BASE_URL + link_tag['href'])
    except requests.exceptions.RequestException as e:
        print(f"Помилка при запиті до {list_url}: {e}")
    return links

# --- ОСНОВНА ФУНКЦІЯ ПАРСИНГУ ДЕТАЛЕЙ (повністю переписана згідно вашого аналізу) ---
def scrape_yacht_details(yacht_url):
    try:
        response = requests.get(yacht_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        yacht_data = {}

        # Назва (очищена)
        h1_tag = soup.find('h1')
        if h1_tag:
            full_name = h1_tag.text.strip()
            yacht_data['Name'] = re.sub(r'\s+YACHT.*', '', full_name, flags=re.IGNORECASE).strip()
        else:
            yacht_data['Name'] = 'N/A'
        
        # Короткий опис
        intro_p = soup.find('p', class_='yacht-intro')
        yacht_data['Description'] = intro_p.text.strip() if intro_p else 'N/A'

        # Блок "Quick View" (Гості, Каюти, Екіпаж)
        quick_view = soup.find('div', class_='quick-view')
        if quick_view:
            for li in quick_view.select('ul.accomodation > li'):
                heading = li.find('p', class_='heading')
                number = li.find('p', class_='number')
                if heading and number:
                    yacht_data[heading.text.strip()] = number.text.strip()
            
            # Конфігурація кают
            cabin_config_div = quick_view.find('div', class_='accomodation-details')
            if cabin_config_div:
                cabin_items = [li.text.strip() for li in cabin_config_div.select('ul > li')]
                yacht_data['Cabin Configuration'] = ', '.join(cabin_items)

        # Таблиця специфікацій
        spec_table = soup.find('table', class_='minimal-style')
        if spec_table:
            for row in spec_table.find_all('tr'):
                title_cell = row.find('td', class_='title')
                value_cell = title_cell.find_next_sibling('td') if title_cell else None
                if title_cell and value_cell:
                    key = title_cell.text.strip()
                    value = ' '.join(value_cell.text.split())
                    yacht_data[key] = value
        
        # Блок з цінами та регіонами по сезонах
        rates_brochure = soup.find('div', class_='yacht-brochure')
        if rates_brochure:
            for season_block in rates_brochure.find_all('div', class_='rates-block'):
                season_heading = season_block.find('p', class_='heading')
                if not season_heading: continue
                
                season_name = season_heading.text.strip().replace(" Season", "") # Summer / Winter
                
                # Ціни (низький та високий сезон)
                prices = season_block.find_all('p', class_='price')
                if len(prices) > 0:
                    low_season_price_raw = ' '.join(prices[0].text.split())
                    yacht_data[f'{season_name} Low Season Price per day $'] = low_season_price_raw
                if len(prices) > 1:
                    high_season_price_raw = ' '.join(prices[1].text.split())
                    yacht_data[f'{season_name} High Season Price per day $'] = high_season_price_raw
                
                # Регіони круїзу
                region_list = season_block.find('p', class_='region-list')
                if region_list:
                    regions = [a.text.strip() for a in region_list.find_all('a')]
                    yacht_data[f'{season_name} Cruising Regions'] = ', '.join(regions)
                
                # Hot Spots
                hot_spots_p = season_block.find('p', class_='hot-spots')
                if hot_spots_p:
                    spots = [a.text.strip().replace(',', '') for a in hot_spots_p.find_all('a')]
                    yacht_data[f'{season_name} Hot Spots'] = ', '.join(spots)
            
        return yacht_data

    except requests.exceptions.RequestException as e:
        print(f"Не вдалося завантажити сторінку {yacht_url}: {e}")
    except Exception as e:
        print(f"Помилка при парсингу сторінки {yacht_url}: {e}")
    return None

# --- ГОЛОВНИЙ СКРИПТ (залишається без змін) ---
if __name__ == "__main__":
    all_yachts_data = []

    for type_id, type_name in YACHT_TYPES.items():
        print(f"\n{'='*30}\n scraping Category: {type_name} (ID: {type_id})\n{'='*30}")
        
        for page_num in range(1, PAGES_TO_SCRAPE_PER_TYPE + 1):
            url = f"{START_URL}?page={page_num}&yacht_type_id_list={type_id}&sort_by=relevance"
            print(f"Обробка сторінки: {url}")
            
            links_on_page = get_yacht_links(url)
            
            if not links_on_page:
                print(f"На сторінці {page_num} не знайдено яхт. Переходимо до наступної категорії.")
                break 
            
            for link in links_on_page:
                print(f"  -> Збираємо дані: {link}")
                details = scrape_yacht_details(link)
                if details:
                    details['Type'] = type_name
                    all_yachts_data.append(details)
                time.sleep(1) 
        
        print(f"Завершено збір даних для категорії '{type_name}'.")
        time.sleep(2) 

    if all_yachts_data:
        print("\nЗбереження всіх зібраних даних у CSV файл...")
        df = pd.DataFrame(all_yachts_data)
        
        # Визначаємо бажаний порядок стовпців
        ordered_columns = [
            'Name', 'Type', 'Guests', 'Cabins', 'Crew', 'Cabin Configuration', 
            'Length', 'Builder', 'Built', 'Cruising Speed', 'Beam', 'Draft', 
            'Gross Tonnage', 'Model', 'Exterior Designer', 'Interior Design',
            'Summer Low Season Price', 'Summer High Season Price', 'Summer Cruising Regions', 'Summer Hot Spots',
            'Winter Low Season Price', 'Winter High Season Price', 'Winter Cruising Regions', 'Winter Hot Spots',
            'Description'
        ]
        
        existing_columns = [col for col in ordered_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in existing_columns]
        final_columns = existing_columns + other_columns
        
        df_ordered = df[final_columns]
        
        df_ordered.to_csv('yachts_data.csv', index=False, encoding='utf-8-sig')
        print(f"✅ Готово! Зібрано дані по {len(all_yachts_data)} яхтам. Результат у файлі 'yachts_data_final_and_correct.csv'.")
    else:
        print("Не вдалося зібрати жодних даних.")
    
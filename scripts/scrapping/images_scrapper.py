import requests
from bs4 import BeautifulSoup
import os
import re
import time

# --- НАЛАШТУВАННЯ ---

BASE_URL = "https://www.yachtcharterfleet.com"
START_URL = "https://www.yachtcharterfleet.com/charter/superyachts-for-charter"
# Кількість сторінок для скрапінгу ДЛЯ КОЖНОГО ТИПУ яхт.
# Поставте 1 для тестування, потім можете збільшити.
PAGES_TO_SCRAPE_PER_TYPE = 42
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Головна папка для збереження всіх зображень
IMAGE_BASE_DIR = "yachts"

YACHT_TYPES = {
    1: 'Motor Yachts', 2: 'Sailing Yachts', 3: 'Expedition Yachts',
    4: 'Classic Yachts', 5: 'Open Yachts', 6: 'Catamarans',
    7: 'Sport Fishing', 8: 'Gulet Yachts'
}

# --- ОСНОВНІ ФУНКЦІЇ ---

def get_yacht_links(list_url):
    """Збирає посилання на сторінки окремих яхт зі сторінки списку."""
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

def download_image(url, folder_path, filename):
    """
    Завантажує одне зображення за URL і зберігає його.
    (ОНОВЛЕНО: Пропускає, якщо файл вже існує)
    """
    full_path = os.path.join(folder_path, filename)
    
    # --- ГОЛОВНА ЗМІНА ---
    # Перевіряємо, чи файл вже існує за цим шляхом
    if os.path.exists(full_path):
        print(f"      ✅ Вже існує: {filename}")
        return # Негайно виходимо з функції
    # ---------------------

    try:
        img_response = requests.get(url, stream=True, headers=HEADERS, timeout=15)
        img_response.raise_for_status()
        
        # Запобігаємо збереженню HTML-сторінок помилок як зображень
        content_type = img_response.headers.get('Content-Type')
        if 'image' not in content_type:
            print(f"      ❌ Помилка: URL не є зображенням ({content_type}): {url}")
            return

        with open(full_path, 'wb') as f:
            for chunk in img_response.iter_content(8192):
                f.write(chunk)
        print(f"      ✅ Збережено нове: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"      ❌ Помилка завантаження {url}: {e}")

def download_images_for_yacht(yacht_url):
    """
    Знаходить і завантажує всі зображення для однієї яхти.
    (ОНОВЛЕНО: Шукає в ДВОХ можливих контейнерах для додаткових фото)
    """
    try:
        response = requests.get(yacht_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. Отримуємо назву яхти для створення папки
        h1_tag = soup.find('h1')
        if h1_tag:
            full_name = h1_tag.text.strip()
            yacht_name = re.sub(r'\s+YACHT.*', '', full_name, flags=re.IGNORECASE).strip()
        else:
            yacht_name = f"unknown_yacht_{int(time.time())}"
        
        sanitized_name = re.sub(r'[^\w\s-]', '', yacht_name).strip()
        sanitized_name_upper = sanitized_name.upper()
        
        yacht_folder_path = os.path.join(IMAGE_BASE_DIR, sanitized_name_upper)
        os.makedirs(yacht_folder_path, exist_ok=True)
        # print(f"   Обробка папки: {yacht_folder_path}")

        # 2. Завантажуємо головне зображення (з подвійною логікою)
        main_image_tag = soup.find('img', id='overview_image')
        if not main_image_tag:
            main_image_tag = soup.find('img', attrs={'data-image-name': re.compile(yacht_name, re.IGNORECASE)})

        if main_image_tag and main_image_tag.has_attr('src'):
            main_image_url = main_image_tag['src']
            if not main_image_url.startswith('http'):
                main_image_url = BASE_URL + main_image_url
            print(f"    -> Перевірка головного зображення (00_main.jpg)...")
            download_image(main_image_url, yacht_folder_path, "00_main.jpg")
        else:
            print(f"    ❌ Не знайдено головне зображення для {yacht_name}")

        
        # 3. Завантажуємо додаткові зображення (✅ ВИПРАВЛЕНИЙ СЕЛЕКТОР)
        
        # --- ГОЛОВНА ЗМІНА ---
        # Спочатку шукаємо один тип контейнера
        additional_images_container = soup.find('div', class_='jsReplaceSlidesHereForMobile')
        
        # Якщо його не знайдено, шукаємо другий тип (який ви знайшли)
        if not additional_images_container:
            additional_images_container = soup.find('div', class_='jsTakeImagesFromHere')
        # ---------------------

        if additional_images_container:
            # Тепер шукаємо 'lightbox' ТІЛЬКИ всередині знайденого контейнера
            image_links = additional_images_container.find_all('a', class_='lightbox')
            
            if image_links:
                print(f"    -> Перевірка {len(image_links)} додаткових зображень...")
                for i, link_tag in enumerate(image_links):
                    if link_tag.has_attr('href'):
                        image_url = link_tag['href']
                        
                        if not image_url.startswith('http'):
                            image_url = BASE_URL + image_url
                            
                        file_extension = os.path.splitext(image_url.split('?')[0])[1]
                        if not file_extension or len(file_extension) > 5:
                            file_extension = ".jpg" 
                        
                        filename = f"{i+1:02d}{file_extension}"
                        download_image(image_url, yacht_folder_path, filename)
            else:
                 print("    -> Знайдено контейнер, але в ньому 0 додаткових зображень.")
        else:
            print("    -> Не знайдено контейнер для дод. зображень (ні 'jsReplaceSlidesHereForMobile', ні 'jsTakeImagesFromHere').")
            
    except requests.exceptions.RequestException as e:
        print(f"Не вдалося завантажити сторінку {yacht_url}: {e}")
    except Exception as e:
        print(f"Помилка при обробці сторінки {yacht_url}: {e}")

# --- ГОЛОВНИЙ СКРИПТ ---
if __name__ == "__main__":
    # Створюємо головну папку, якщо її немає
    os.makedirs(IMAGE_BASE_DIR, exist_ok=True)
    print(f"Зображення будуть збережені в папку: '{IMAGE_BASE_DIR}'")

    # Головний цикл по типах яхт
    for type_id, type_name in YACHT_TYPES.items():
        print(f"\n{'='*40}\n scraping Category: {type_name}\n{'='*40}")
        
        # Цикл по сторінках для даного типу
        for page_num in range(1, PAGES_TO_SCRAPE_PER_TYPE + 1):
            url = f"{START_URL}?page={page_num}&yacht_type_id_list={type_id}&sort_by=relevance"
            print(f"Обробка сторінки: {url}")
            
            links_on_page = get_yacht_links(url)
            
            if not links_on_page:
                print(f"На сторінці {page_num} не знайдено яхт. Переходимо до наступної категорії.")
                break 
            
            # Завантажуємо зображення для кожної знайденої яхти
            for link in links_on_page:
                print(f"  Обробка яхти: {link}")
                download_images_for_yacht(link)
                time.sleep(1) # Невелика затримка
        
        print(f"Завершено роботу з категорією '{type_name}'.")
        time.sleep(2)
        
    print("\n\n🎉 Всі операції завершено!")

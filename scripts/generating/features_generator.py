import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime

df = pd.read_csv('yachts_data_cleaned.csv')

def generate_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерує рейтинг яхти на основі її характеристик
    """
    df = df.copy()
    
    # Конвертуємо всі необхідні колонки в числовий формат
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['crew'] = pd.to_numeric(df['crew'], errors='coerce')
    df['guests'] = pd.to_numeric(df['guests'], errors='coerce')
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    df['summer_low_season_price_per_day_$'] = pd.to_numeric(df['summer_low_season_price_per_day_$'], errors='coerce')
    df['summer_high_season_price_per_day_$'] = pd.to_numeric(df['summer_high_season_price_per_day_$'], errors='coerce')
    df['winter_low_season_price_per_day_$'] = pd.to_numeric(df['winter_low_season_price_per_day_$'], errors='coerce')
    df['winter_high_season_price_per_day_$'] = pd.to_numeric(df['winter_high_season_price_per_day_$'], errors='coerce')
    
    # Ініціалізуємо колонку score для всіх рядків
    df['score'] = 0.0
    
    # 1. AGE FACTOR - Newer yachts get higher ratings
    age = 2025 - df['year']
    df.loc[age < 8, 'score'] += 1.0
    df.loc[(age >= 8) & (age < 100), 'score'] += 0.5
    
    # 2. CREW RATIO FACTOR - High crew-to-guest ratio = better service
    # Уникаємо ділення на нуль
    crew_ratio = df['crew'] / df['guests'].replace(0, np.nan)
    df.loc[crew_ratio > 0.5, 'score'] += 1.0
    df.loc[(crew_ratio > 0.3) & (crew_ratio <= 0.5), 'score'] += 0.5
    
    # 3. SIZE FACTOR - Larger yachts = more impressive
    df.loc[df['length'] > 150, 'score'] += 1.0
    df.loc[(df['length'] > 50) & (df['length'] <= 150), 'score'] += 0.5
    
    # 4. PRICE FACTOR - Very expensive = luxury = high rating
    avg_summer_price = (df['summer_low_season_price_per_day_$'] + df['summer_low_season_price_per_day_$']) / 2
    df.loc[avg_summer_price > 375000, 'score'] += 1.0
    df.loc[(avg_summer_price > 125000) & (avg_summer_price <= 375000), 'score'] += 0.5
    
    # 5. ADD NOISE for realism
    noise = np.random.normal(0, 0.3, size=len(df))
    
    # 6. CALCULATE FINAL RATING [3.0 - 5.0]
    df['rating'] = 3.0 + np.minimum(2.0, df['score'] + noise)
    
    # Round to 1 decimal place
    df['rating'] = df['rating'].round(1)
    
    # Ensure rating is within bounds
    df['rating'] = df['rating'].clip(3.0, 5.0)
    
    # Remove temporary score column
    df = df.drop('score', axis=1)
    
    # Handle any NaN ratings (default to 4.0)
    df['rating'] = df['rating'].fillna(4.0)
    
    return df

def add_base_marina(df: pd.DataFrame) -> pd.DataFrame:
    """
    Версія з урахуванням регіону при заповненні пропусків.
    Більш інтелектуальний підхід: марини з того ж регіону.
    """
    df = df.copy()
    
    # Створюємо допоміжну функцію, яка буде обробляти кожен окремий рядок
    def find_base_marina(row):
        s_str = row['Summer Hot Spots']
        w_str = row['Winter Hot Spots']
        summer_list = [p.strip() for p in str(s_str).split(',') if p.strip() and pd.notna(s_str)]
        winter_list = [p.strip() for p in str(w_str).split(',') if p.strip() and pd.notna(w_str)]
        
        if not summer_list and not winter_list:
            return np.nan
        
        if summer_list and winter_list:
            winter_set = set(winter_list)
            common_ports = [port for port in summer_list if port in winter_set]
            if common_ports:
                return common_ports[0]
        
        if summer_list:
            return summer_list[0]
        
        if winter_list:
            return winter_list[0]
        
        return np.nan
    
    df['base_marina'] = df.apply(find_base_marina, axis=1)
    
    # --- ЗАПОВНЕННЯ З УРАХУВАННЯМ РЕГІОНУ ---
    
    # Створюємо мапінг регіон -> марини (з частотами)
    region_marina_mapping = {}
    
    for idx, row in df.iterrows():
        if pd.notna(row['base_marina']):
            # Збираємо регіони
            regions = []
            if 'Summer Cruising Regions' in df.columns and pd.notna(row['Summer Cruising Regions']):
                summer_regions = [r.strip() for r in str(row['Summer Cruising Regions']).split(',')]
                regions.extend(summer_regions)
            if 'Winter Cruising Regions' in df.columns and pd.notna(row['Winter Cruising Regions']):
                winter_regions = [r.strip() for r in str(row['Winter Cruising Regions']).split(',')]
                regions.extend(winter_regions)
            
            marina = row['base_marina']
            
            for region in regions:
                if region:
                    if region not in region_marina_mapping:
                        region_marina_mapping[region] = []
                    region_marina_mapping[region].append(marina)
    
    # Конвертуємо в ймовірності для кожного регіону
    region_marina_probs = {}
    for region, marinas in region_marina_mapping.items():
        if marinas:
            marina_counts = pd.Series(marinas).value_counts()
            region_marina_probs[region] = {
                'marinas': marina_counts.index.tolist(),
                'probs': (marina_counts / marina_counts.sum()).tolist()
            }
    
    # Fallback: загальний розподіл марін
    global_marina_counts = df['base_marina'].dropna().value_counts()
    
    if len(global_marina_counts) > 0:
        global_marina_probs = global_marina_counts / global_marina_counts.sum()
        
        # Заповнюємо пропуски
        missing_indices = df[df['base_marina'].isna()].index
        
        filled_count = 0
        region_based_count = 0
        
        for idx in missing_indices:
            # Спробуємо знайти марину з того ж регіону
            regions = []
            if 'Summer Cruising Regions' in df.columns and pd.notna(df.loc[idx, 'Summer Cruising Regions']):
                summer_regions = [r.strip() for r in str(df.loc[idx, 'Summer Cruising Regions']).split(',')]
                regions.extend(summer_regions)
            
            sampled_marina = None
            
            # Шукаємо марину з того ж регіону
            for region in regions:
                if region in region_marina_probs:
                    probs_data = region_marina_probs[region]
                    sampled_marina = np.random.choice(
                        probs_data['marinas'],
                        p=probs_data['probs']
                    )
                    region_based_count += 1
                    break
            
            # Якщо не знайшли за регіоном, використовуємо загальний розподіл
            if sampled_marina is None:
                sampled_marina = np.random.choice(
                    global_marina_probs.index,
                    p=global_marina_probs.values
                )
            
            df.loc[idx, 'base_marina'] = sampled_marina
            filled_count += 1

    df = df.drop(columns=['Summer Cruising Regions', 'Summer Hot Spots', 'Winter Cruising Regions', 'Winter Hot Spots'])
    
    return df

def add_country_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Створює нову колонку 'Country' на основі значень у колонці 'Base Marina'.
    """
    df = df.copy()

    # Створюємо словник відповідності "Марина -> Країна"
    marina_to_country_map = {
        # Італія
        'Amalfi Coast': 'Italy',
        'Sardinia': 'Italy',
        'Ligurian Riviera': 'Italy',
        
        # Греція
        'Mykonos': 'Greece',
        
        # Франція
        'Calvi': 'France',
        'Corsica': 'France',
        'French Riviera': 'France',
        'Cannes': 'France',
        
        # Іспанія
        'Ibiza': 'Spain',
        'Mallorca': 'Spain',
        'The Balearics': 'Spain',
        
        # США
        'Virgin Islands': 'USA',
        'Alaska': 'USA',
        'Florida': 'USA',
        'New England': 'USA',
        
        # ОАЕ
        'Abu Dhabi': 'UAE',
        'Dubai': 'UAE',
        
        # Австралія
        'Sydney': 'Australia',
        'Whitsundays': 'Australia',
        
        # Інші
        'Komodo': 'Indonesia',
        
        # Регіони (не країни)
        'Scandinavia': 'Norway'
    }

    # Використовуємо .map() для створення нової колонки
    # Якщо 'Base Marina' немає в словнику, буде поставлено NaN (порожнє)
    df['country'] = df['base_marina'].map(marina_to_country_map)
    
    return df

DEFAULT_PHOTO_URL = "https://pub-59edec60055841149d71125f2e73e658.r2.dev/yachts/yachts/No Image.jpg"

def format_for_postgres_array(url_list) -> str:
    """
    Конвертує список Python ['a', 'b'] у рядок масиву Postgres '{"a","b"}'.
    Якщо список порожній або NaN, повертає масив з одним URL-заглушкою.
    """

    if not isinstance(url_list, list) or not url_list:
        # Якщо так, повертаємо масив з одним посиланням-заглушкою
        return '{"' + DEFAULT_PHOTO_URL + '"}'
    
    # Якщо список валідний (не порожній), обробляємо його
    cleaned_urls = [str(url).replace('"', '""') for url in url_list]
    return '{"' + '","'.join(cleaned_urls) + '"}'

# --- 3. Головна функція (залишається без змін, але я додаю її для контексту) ---
def add_photo_urls(df: pd.DataFrame, photo_map_file: str) -> pd.DataFrame:
    """
    Створює колонку 'Photos' у DataFrame, використовуючи карту з JSON-файлу.
    Додає URL-заглушку, якщо фото не знайдено.
    """
    df = df.copy()
    
    default_array_str = format_for_postgres_array(np.nan)

    try:
        with open(photo_map_file, 'r') as f:
            photo_map = json.load(f)
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{photo_map_file}' не знайдено!")
        print("Заповнюю колонку 'Photos' посиланнями-заглушками.")
        df['photos'] = default_array_str
        return df

    photo_lists = df['name'].map(photo_map)

    # Тепер ця .apply() буде працювати коректно
    df['photos'] = photo_lists.apply(format_for_postgres_array)
    
    return df

def drop_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє рядки, де всі чотири вказані колонки цін дорівнюють 0.
    """
    price_columns = [
        'summer_low_season_price_per_day_$', 
        'summer_high_season_price_per_day_$', 
        'winter_low_season_price_per_day_$', 
        'winter_high_season_price_per_day_$'
    ]
    
    # Переконуємося, що колонки є числовими, і заповнюємо NaN нулями 
    # для коректного порівняння
    df_copy = df.copy()
    for col in price_columns:
        if col in df_copy.columns:
            # errors='coerce' перетворить будь-які нечислові дані на NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        else:
            print(f"Попередження: Колонка '{col}' не знайдена у DataFrame.")

    # 1. Створюємо DataFrame з True/False (True, якщо значення 0)
    is_zero = (df_copy[price_columns] == 0)
    
    # 2. Створюємо маску (True, якщо ВСІ 4 колонки були True)
    mask_all_zeroes = is_zero.all(axis=1)

    # 3. Фільтруємо: залишаємо тільки ті рядки, де маска НЕ True
    df_filtered = df_copy[~mask_all_zeroes]
    
    print(f"Видалено {len(df_copy) - len(df_filtered)} рядків з нульовими цінами.")
    
    return df_filtered

def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Назви колонок
    sl = 'summer_low_season_price_per_day_$'
    sh = 'summer_high_season_price_per_day_$'
    wl = 'winter_low_season_price_per_day_$'
    wh = 'winter_high_season_price_per_day_$'
    
    price_cols = [sl, sh, wl, wh]

    # Переконуємось, що всі колонки числові
    original_dtypes = {}
    for col in price_cols:
        if col in df.columns:
            original_dtypes[col] = df[col].dtype
            # errors='coerce' перетворить будь-які нечислові дані на NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            print(f"ПОПЕРЕДЖЕННЯ: Колонка '{col}' не знайдена і буде проігнорована.")
            price_cols.remove(col)

    print("--- Ціни ДО заповнення (приклад) ---")
    print(df[price_cols].head())

    # --- ЕТАП 1: Заповнення Summer Low (SL) з інших колонок ---
    
    # Якщо SL=0, спробувати взяти з SH
    mask_sl_from_sh = (df[sl] == 0) & (df[sh] > 0)
    df.loc[mask_sl_from_sh, sl] = df.loc[mask_sl_from_sh, sh]
    
    # Якщо SL *все ще* 0, спробувати взяти з WL
    mask_sl_from_wl = (df[sl] == 0) & (df[wl] > 0)
    df.loc[mask_sl_from_wl, sl] = df.loc[mask_sl_from_wl, wl]
    
    # Якщо SL *все ще* 0, спробувати взяти з WH
    mask_sl_from_wh = (df[sl] == 0) & (df[wh] > 0)
    df.loc[mask_sl_from_wh, sl] = df.loc[mask_sl_from_wh, wh]

    # --- ЕТАП 2: Прямий каскад від (тепер заповненого) SL ---

    # Правило 2.1: Заповнити Summer High з Summer Low
    mask_sh_from_sl = (df[sh] == 0) & (df[sl] > 0)
    df.loc[mask_sh_from_sl, sh] = df.loc[mask_sh_from_sl, sl]

    # Правило 2.2: Заповнити Winter Low з Summer Low
    mask_wl_from_sl = (df[wl] == 0) & (df[sl] > 0)
    df.loc[mask_wl_from_sl, wl] = df.loc[mask_wl_from_sl, sl]

    # --- ЕТАП 3: Заповнення Winter High (WH) ---
    
    # Правило 3.1: Заповнити Winter High з Winter Low (який міг бути щойно заповнений)
    mask_wh_from_wl = (df[wh] == 0) & (df[wl] > 0)
    df.loc[mask_wh_from_wl, wh] = df.loc[mask_wh_from_wl, wl]
    
    # Правило 3.2: Запасний варіант, заповнити Winter High з Summer High
    mask_wh_from_sh = (df[wh] == 0) & (df[sh] > 0)
    df.loc[mask_wh_from_sh, wh] = df.loc[mask_wh_from_sh, sh]

    print("\n--- Ціни ПІСЛЯ заповнення (приклад) ---")
    print(df[price_cols].head())
    
    # Повертаємо оригінальні типи даних (наприклад, int, якщо вони були int)
    for col in price_cols:
        df[col] = df[col].astype(original_dtypes[col])

    return df

df = df.drop(columns=['Unnamed: 0'])

df['name'] = df['name'].str.strip()

df = df.pipe(generate_rating).pipe(add_base_marina).pipe(add_country_column).pipe(add_photo_urls, photo_map_file='photo_map.json').pipe(drop_zero_prices).pipe(fill_missing_prices)

df['userId'] = 'ff210a03-d01e-49a5-9050-284d1d94490a'

df['createdAt'] = datetime.now()

# df.to_csv('yachts_data_filled.csv')

# print(df.isna().sum())

# print(df[['summer_low_season_price_per_day_$', 'summer_high_season_price_per_day_$', 'winter_low_season_price_per_day_$', 'winter_high_season_price_per_day_$']].describe())

print(df['summer_low_season_price_per_day_$'].describe())

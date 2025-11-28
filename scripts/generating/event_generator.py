import pandas as pd
from faker import Faker
import random
import uuid
from datetime import datetime, timedelta
import numpy as np # Потрібен для роботи з NaN

# --- Налаштування ---
N_USERS = 501      # Скільки користувачів згенерувати
N_EVENTS = 12525   # Скільки подій згенерувати
# --------------------

# Ініціалізуємо Faker
fake = Faker()

# --- 0. Завантаження та підготовка даних про яхти ---
# Нам потрібні не тільки ID, але й ціни та країни для розрахунків
try:
    # Завантажуємо ПОВНИЙ файл з даними
    yacht_df = pd.read_csv('yachts_202511181125.csv') 
    
    # --- Нова логіка: Обробка цін яхт ---
    price_cols = ['summerLowSeasonPrice', 'summerHighSeasonPrice', 'winterLowSeasonPrice', 'winterHighSeasonPrice']
    
    # Переконуємось, що ціни є числовими, ігноруючи помилки
    for col in price_cols:
        yacht_df[col] = pd.to_numeric(yacht_df[col], errors='coerce')
        
    # Створюємо середню ціну яхти, ігноруючи пропуски (NaN)
    yacht_df['avg_yacht_price'] = yacht_df[price_cols].mean(axis=1, skipna=True)
    
    # --- Нова логіка: Перевірка країни ---
    if 'country' not in yacht_df.columns:
        print("ПОПЕРЕДЖЕННЯ: У файлі 'yachts_202510301923.csv' немає колонки 'country'.")
        print("Логіка Етапу 2 (Бюджет + Країна) не зможе працювати коректно.")
        # Створюємо фіктивну колонку, щоб скрипт не впав
        yacht_df['country'] = None 
        
    # Створюємо DataFrame для швидкого пошуку яхт (видаляємо яхти без ціни)
    yachts_lookup_df = yacht_df[['id', 'avg_yacht_price', 'country']].dropna(subset=['avg_yacht_price'])
    YACHT_IDS = yacht_df['id'].tolist() # Повний список ID для "запасного" варіанту

    print(f"Знайдено та оброблено {len(YACHT_IDS)} яхт.")
    
except FileNotFoundError:
    print("ПОМИЛКА: Файл 'yachts_data_filled.csv' не знайдено.")
    print("Використовую фіктивні ID яхт.")
    YACHT_IDS = [f'yacht_id_{i}' for i in range(1, 501)]
    yachts_lookup_df = None # Вимикаємо нову логіку


# Словник типів подій та їхньої "ваги" (для запису в БД)
EVENT_TYPES = {
    'view': 2,
    'wishlist': 4,
    'chatOwner': 6,
    'startBooking': 8,
    'book': 10
}

# Нові ваги для ГЕНЕРАЦІЇ (воронка)
GENERATION_WEIGHTS = {
    'view': 70,
    'wishlist': 18,
    'chatOwner': 6,
    'startBooking': 3.5,
    'book': 2.5
}

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
    'Scandinavia': 'Norway'
}
SAILING_EXP_OPTIONS = ['none', 'beginner', 'intermediate', 'pro']

if yachts_lookup_df is not None:
    # Cтворюємо список реальних середніх цін на яхти
    available_yacht_prices = yachts_lookup_df['avg_yacht_price'].tolist()

print("Починаємо генерацію користувачів...")

# --- 1. Генерація користувачів (Users) ---
# (Цей блок не змінено, окрім виправлення помилки .round())
users_data = []
for _ in range(N_USERS):
    user_id = str(uuid.uuid4())
    created_at = fake.date_time_between(start_date=datetime(2025, 1, 1), end_date='now')
    
    # ВИПРАВЛЕННЯ: .round(-3) - це метод pandas, а не int. 
    # Використовуємо // для округлення до тисяч.
    base_price = random.choice(available_yacht_prices)
    
    # 2. Додаємо випадковий "зсув" +/- 10% (як ви просили)
    shift = random.uniform(-0.10, 0.10)
    budget_min_raw = base_price * (1 + shift)
    
    # 3. Округлюємо до тисяч
    budget_min = int((budget_min_raw // 1000) * 1000)
    
    # 4. Встановлюємо абсолютний мінімум
    if budget_min < 1000:
        budget_min = 1000
    
    # --- Нова логіка для budget_max ---
    
    # 1. Визначаємо випадковий відсоток збільшення (від 0% до 50%)
    percentage_increase = random.uniform(0.0, 0.50) 
    
    # 2. Розраховуємо budget_max (min + min * відсоток)
    budget_max_raw = budget_min * (1 + percentage_increase)
    
    # 3. Округлюємо budget_max до тисяч, так само, як і budget_min
    budget_max = int((budget_max_raw // 1000) * 1000)
    
    # 4. Перевірка: через округлення вниз budget_max може стати 
    #    меншим за budget_min. Цей рядок це виправляє.
    if budget_max < budget_min:
        budget_max = budget_min

    user = {
        'id': user_id,
        'email': fake.unique.email(),
        'password_hash': fake.sha256(),
        'country': random.choice(list(set(marina_to_country_map.values()))),
        'role': 'lessee',
        'sailingExp': random.choice(SAILING_EXP_OPTIONS),
        'budgetMin': budget_min,
        'budgetMax': budget_max,
        'has_skipper_licence': random.choice([True, False]),
        'createdAt': created_at,
        'updatedAt': created_at
    }
    users_data.append(user)

users_df = pd.DataFrame(users_data)
users_df.to_csv('generated_users.csv', index=False)
print(f"✅ Створено {len(users_df)} користувачів у 'generated_users.csv'")


# --- 2. Нова підготовка до генерації подій ---

# Створюємо середній бюджет для кожного користувача для швидкого доступу
users_lookup_df = users_df.copy()
users_lookup_df['budgetMin'] = pd.to_numeric(users_lookup_df['budgetMin'], errors='coerce')
users_lookup_df['budgetMax'] = pd.to_numeric(users_lookup_df['budgetMax'], errors='coerce')
users_lookup_df['avg_user_budget'] = users_lookup_df[['budgetMin', 'budgetMax']].mean(axis=1, skipna=True)
users_lookup_df = users_lookup_df[['id', 'country', 'avg_user_budget', 'createdAt']]

# Створюємо словник для миттєвого доступу до даних юзера
users_dict = users_lookup_df.set_index('id').to_dict('index')

# Створюємо списки для зваженого випадкового вибору типу події
event_names = list(EVENT_TYPES.keys())
generation_weights_list = [GENERATION_WEIGHTS[name] for name in event_names]

events_data = []
user_ids_list = users_df['id'].tolist()

# Визначаємо кількість подій для кожного етапу
n_step1_events = int(N_EVENTS * 0.20)
n_step2_events = N_EVENTS - n_step1_events

print(f"\nПочинаємо генерацію {N_EVENTS} подій...")

# --- 3. Генерація подій (Events) ---

# --- Етап 1: 20% подій (на основі бюджету) ---
print(f"Генеруємо {n_step1_events} подій (Етап 1: Бюджет +/- 10%)...")
for _ in range(n_step1_events):
    user_id = random.choice(user_ids_list)
    user_info = users_dict[user_id]
    user_budget = user_info['avg_user_budget']
    
    yacht_id = None
    
    # Перевіряємо, чи є в нас дані для цієї логіки
    if yachts_lookup_df is not None and pd.notna(user_budget):
        budget_min_limit = user_budget * 0.9  # -10%
        budget_max_limit = user_budget * 1.1  # +10%
        
        # Знаходимо яхти, що відповідають бюджету
        matching_yachts = yachts_lookup_df[
            (yachts_lookup_df['avg_yacht_price'] >= budget_min_limit) &
            (yachts_lookup_df['avg_yacht_price'] <= budget_max_limit)
        ]
        
        if not matching_yachts.empty:
            yacht_id = matching_yachts.sample(1)['id'].iloc[0]
    
    # "Запасний" варіант: якщо логіка не спрацювала (напр. немає бюджету), беремо будь-яку яхту
    if yacht_id is None:
        yacht_id = random.choice(YACHT_IDS)
        
    # Генеруємо саму подію
    event_type = random.choices(event_names, weights=generation_weights_list, k=1)[0]
    user_created_at = user_info['createdAt']
    event_time = fake.date_time_between(start_date=user_created_at, end_date='now')
    
    event = {
        'id': str(uuid.uuid4()),
        'userId': user_id,
        'yachtId': yacht_id, # <-- Зберігаємо ТІЛЬКИ ID, як ви й просили
        'type': event_type,
        'weight': EVENT_TYPES[event_type],
        'createdAt': event_time,
        'updatedAt': event_time # createdAt має дорівнювати updatedAt
    }
    events_data.append(event)

# --- Етап 2: 80% подій (Бюджет + Країна) ---
print(f"Генеруємо {n_step2_events} подій (Етап 2: Бюджет +/- 10% + Країна)...")
for _ in range(n_step2_events):
    user_id = random.choice(user_ids_list)
    user_info = users_dict[user_id]
    user_budget = user_info['avg_user_budget']
    user_country = user_info['country']

    yacht_id = None

    # Перевіряємо, чи є в нас дані для цієї логіки
    if yachts_lookup_df is not None and pd.notna(user_budget) and pd.notna(user_country):
        budget_min_limit = user_budget * 0.9
        budget_max_limit = user_budget * 1.1

        # Шукаємо яхти, що збігаються І за бюджетом, І за країною
        budget_match = (yachts_lookup_df['avg_yacht_price'] >= budget_min_limit) & (yachts_lookup_df['avg_yacht_price'] <= budget_max_limit)
        country_match = (yachts_lookup_df['country'] == user_country)
        
        matching_yachts = yachts_lookup_df[budget_match & country_match]
        
        if not matching_yachts.empty:
            yacht_id = matching_yachts.sample(1)['id'].iloc[0]
        else:
            # "Запасний" варіант 2.1: Шукаємо ТІЛЬКИ за бюджетом (якщо країна не збіглася)
            matching_yachts_budget_only = yachts_lookup_df[budget_match]
            if not matching_yachts_budget_only.empty:
                yacht_id = matching_yachts_budget_only.sample(1)['id'].iloc[0]

    # "Запасний" варіант 2.2: якщо нічого не знайдено, беремо будь-яку яхту
    if yacht_id is None:
        yacht_id = random.choice(YACHT_IDS)
        
    # Генеруємо саму подію
    event_type = random.choices(event_names, weights=generation_weights_list, k=1)[0]
    user_created_at = user_info['createdAt']
    event_time = fake.date_time_between(start_date=user_created_at, end_date='now')

    event = {
        'id': str(uuid.uuid4()),
        'userId': user_id,
        'yachtId': yacht_id, # <-- Зберігаємо ТІЛЬКИ ID
        'type': event_type,
        'weight': EVENT_TYPES[event_type],
        'createdAt': event_time,
        'updatedAt': event_time
    }
    events_data.append(event)
    
# --- 4. Збереження подій ---
events_df = pd.DataFrame(events_data)
# ВИПРАВЛЕННЯ: сортуємо за 'createdAt', а не 'ts' (якої не існувало)
events_df = events_df.sort_values(by='createdAt') 

events_df.to_csv('generated_events.csv', index=False)
print(f"✅ Створено {len(events_df)} подій у 'generated_events.csv'")

# --- 5. (Опціонально) Перевірка розподілу подій ---
print("\n--- Перевірка розподілу згенерованих подій ---")
print(events_df['type'].value_counts(normalize=True).sort_index() * 100)

print("\n🎉 Генерація завершена!")

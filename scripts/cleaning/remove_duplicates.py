import pandas as pd
import numpy as np

def deduplicate_by_rarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє дублікати за колонками 'Name' та 'Description', 
    залишаючи той рядок, у якого 'Type' є рідкіснішим (менш поширеним).
    """
    print("--- Початок дедуплікації за рідкістю типу ---")
    
    initial_count = len(df)
    
    # 1. Відкладаємо рядки з NaN в ключових колонках, оскільки ми не можемо їх порівнювати
    key_cols = ['name', 'description', 'type']
    df_clean = df.dropna(subset=key_cols).copy()
    df_to_keep_as_is = df[df[key_cols].isna().any(axis=1)]
    
    print(f"Рядків для обробки (без NaN в ключових колонках): {len(df_clean)}")

    # 2. Розраховуємо глобальну частоту типів (на основі всіх даних)
    type_frequencies = df['type'].value_counts()
    print("\n--- Глобальна частота типів ---")
    print(type_frequencies)

    # 3. Додаємо частоту до кожного рядка в очищеному наборі
    df_clean['type_frequency'] = df_clean['type'].map(type_frequencies)

    # 4. Сортуємо: 
    #    - Спочатку за 'Name' та 'Description', щоб згрупувати дублікати
    #    - Потім за 'Type_Frequency' (ascending - за зростанням).
    # Це гарантує, що "найрідкісніший" тип буде першим у кожній групі дублікатів.
    df_sorted = df_clean.sort_values(
        by=['name', 'description', 'type_frequency'], 
        ascending=[True, True, True]
    )

    # 5. Видаляємо дублікати за 'Name' + 'Description', залишаючи перший (найрідкісніший)
    df_deduplicated = df_sorted.drop_duplicates(subset=['name', 'description'], keep='first')
    
    # 6. Повертаємо рядки з NaN і поєднуємо їх
    df_final = pd.concat([df_to_keep_as_is, df_deduplicated], ignore_index=True)
    
    # 7. Видаляємо тимчасову колонку
    df_final = df_final.drop(columns=['type_frequency'], errors='ignore')

    print(f"\n--- Дедуплікацію завершено ---")
    print(f"Видалено {initial_count - len(df_final)} дублікатів.")
    print(f"Фінальна кількість рядків: {len(df_final)}")
    
    return df_final

# --- ЯК ВИКОРИСТОВУВАТИ ---
if __name__ == "__main__":
    
    file_name = 'yachts_data.csv' # <-- Перевірте назву
    output_file_name = 'yachts_data_deduplicated.csv'

    try:
        # 1. Завантажте ваш DataFrame
        df = pd.read_csv(file_name)
        print(f"Початкова кількість рядків: {len(df)}")
        
        # 2. Застосуйте функцію (можна через .pipe())
        final_df = df.pipe(deduplicate_by_rarity)
        
        # 3. Збережіть результат
        final_df.to_csv(output_file_name, index=True)
        print(f"✅ Успішно збережено у файл: {output_file_name}")

        # duplicate_mask = final_df.duplicated(subset=['name'], keep=False)
        # print(duplicate_mask.sum(), "рядків мають дублікати за 'name' після дедуплікації.")

    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{file_name}' не знайдено.")
    except KeyError as e:
        print(f"ПОМИЛКА: У вашому файлі відсутня необхідна колонка: {e}")

import pandas as pd

df = pd.read_csv('yachts_data_deduplicated.csv')

def delete_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()

   columns_to_drop = ['Cabin Configuration', 'Builder', 'Cruising Speed', 'Beam', 'Draft', 'Gross Tonnage', 'Exterior Designer', 'Interior Design', 'Owner & Guests']
   df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
   return df


def year_cleaner(df: pd.DataFrame) -> pd.DataFrame:
   col_name = 'year'
    
   if col_name in df.columns:
      df = df.copy()

      s = df[col_name].astype(str)

      all_years = s.str.findall(r'(\d{4})')

      last_year = all_years.str[-1]

      df[col_name] = pd.to_numeric(last_year, errors='coerce').astype('Int64')

   return df

def clean_duplicated_words(text: str) -> str:
    if not isinstance(text, str):
        return text

    words = text.split()
    n = len(words)
    
    if n > 0 and n % 2 == 0:
        midpoint = n // 2
        if words[:midpoint] == words[midpoint:]:
            return ' '.join(words[:midpoint])
            
    return text

# Основна функція, тепер складається лише з одного кроку
def model_cleaner(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()

   # Застосовуємо нашу універсальну функцію до кожного рядка
   df['model'] = df['model'].apply(clean_duplicated_words)

   return df

def price_cleaner(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()
    
   col_name = ['summer_low_season_price_per_day_$', 'summer_high_season_price_per_day_$', 'winter_low_season_price_per_day_$', 'winter_high_season_price_per_day_$']
    
   for col in col_name:
      if col in df.columns:
        poa_mask = df[col].astype(str).str.contains('POA', na=False)
        
        df.loc[poa_mask, col] = 0
        
        cleaned_prices = df.loc[~poa_mask, col].str.split(' ').str[-1].str[1:].str.replace(',', '')
        
        df.loc[~poa_mask, col] = pd.to_numeric(cleaned_prices, errors='coerce')
        
        df[col] = df[col].fillna(0).astype(int)

        df[col] = round(df[col] / 7, 0).astype(int)

   return df

def length_cleaner(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()

   df['length'] = df['length'].str.split(' ').str[0].str[:-1].astype(float)

   return df

def regions_cleaner(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()

   df['Summer Cruising Regions'] = df['Summer Cruising Regions'].str.replace(',,', ',')
   df['Winter Cruising Regions'] = df['Winter Cruising Regions'].str.replace(',,', ',')

   return df

df = df.drop(columns=['Unnamed: 0'])

required_columns = ['crew', 'cabins', 'guests', 'year']

df = df.dropna(subset=required_columns)

df['name'] = df['name'].str.strip().str.replace(r'[^a-zA-Z0-9ÀÈÉ²ПЕТРОПАВЛОВСК \-]', '', regex=True)

df = df.pipe(delete_unused_columns).pipe(length_cleaner).pipe(year_cleaner).pipe(model_cleaner).pipe(price_cleaner).pipe(regions_cleaner)

df.to_csv('yachts_data_cleaned.csv')

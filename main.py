from fastapi import FastAPI
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Cargar los archivos parquet una sola vez al inicio
archivos_parquet = ['clean_df_ur.parquet', 'definitivo_df_ui.parquet', 'definitivo_df_sg.parquet']
df_ur = pd.read_parquet('DataSet/' + archivos_parquet[0])
df_ui = pd.read_parquet('DataSet/' + archivos_parquet[1])
df_sg = pd.read_parquet('DataSet/' + archivos_parquet[2])

# Preparar los datos para el modelo de recomendación
combined_df = pd.merge(df_ui[['item_id', 'playtime_forever_log']], df_sg[['id', 'price']], left_on='item_id', right_on='id')
game_features = combined_df[['playtime_forever_log', 'price']]
pca = PCA(n_components=2)
game_features_reduced = pca.fit_transform(game_features)

# Recomendación de juegos basada en PCA y similitud del coseno
def recomendacion_juego(game_id, top_n=5):
    # Asegurar que el game_id sea un índice válido en game_features_reduced
    if game_id not in range(len(game_features_reduced)):
        return "ID de juego no válido"
    
    game_vector = game_features_reduced[game_id:game_id+1]
    cosine_similarities = cosine_similarity(game_vector, game_features_reduced)
    similar_games_indices = cosine_similarities.argsort()[0][-top_n-1:-1][::-1]
    similar_games_indices = [i for i in similar_games_indices if i != game_id]
    recommended_games = df_ui.loc[similar_games_indices, 'item_name']
    return recommended_games[:top_n].tolist()

@app.get("/recomendacion-juego/{game_id}")
async def get_recomendacion_juego(game_id: int):
    recomendaciones = recomendacion_juego(game_id, top_n=5)
    return recomendaciones

# Endpoints funciones
def PlayTimeGenre(genero: str):
    genero = genero.lower()  # Convertir el género ingresado a minúsculas
    merged_df = pd.merge(df_ui, df_sg, left_on='item_id', right_on='id')

    # Filtrar por género y calcular las horas jugadas por año
    filtered_df = merged_df[merged_df['genres'].str.lower().str.contains(genero)]  # Convertir géneros a minúsculas
    hours_by_year = filtered_df.groupby('release_year')['playtime_forever'].sum()
    max_year = hours_by_year.idxmax()

    return {"Año de lanzamiento con más horas jugadas para Género {}".format(genero): max_year}

@app.get("/playtime-genre/{genero}")
async def playtime_genre(genero: str):
    return PlayTimeGenre(genero)

def UserForGenre(genero: str):
    genero = genero.lower()  # Convertir el género ingresado a minúsculas
    merged_df = pd.merge(df_ui, df_sg, left_on='item_id', right_on='id')

    # Filtrar por género
    filtered_df = merged_df[merged_df['genres'].str.lower().str.contains(genero)]  # Convertir géneros a minúsculas
    top_user = filtered_df.groupby('user_id')['playtime_forever'].sum().idxmax()
    hours_by_year = filtered_df[filtered_df['user_id'] == top_user].groupby('release_year')['playtime_forever'].sum().reset_index()
    hours_by_year['playtime_forever'] = hours_by_year['playtime_forever'] / 60

    return {"Usuario con más horas jugadas para Género {}".format(genero): top_user, "Horas jugadas": hours_by_year.to_dict('records')}

@app.get("/user-for-genre/{genero}")
async def user_for_genre(genero: str):
    return UserForGenre(genero)

def UsersRecommend(año: int):
    # Filtrar por año y recomendación positiva/neutra
    filtered_df = df_ur[(df_ur['posted_year'] == año) & (df_ur['recommend'] == True) & (df_ur['sentiment_analysis'] > 0)]

    # Agrupar por juego y contar recomendaciones
    top_games = filtered_df['item_id'].value_counts().head(3).index.tolist()

    return [{"Puesto {}".format(i+1): game} for i, game in enumerate(top_games)]

@app.get("/users-recommend/{año}")
async def users_recommend(año: int):
    return UsersRecommend(año)

def UsersWorstDeveloper(año: int):
    # Unir df_ur con df_sg
    merged_df = pd.merge(df_ur, df_sg, left_on='item_id', right_on='id')

    # Filtrar por año y recomendación negativa
    filtered_df = merged_df[(merged_df['posted_year'] == año) & (merged_df['recommend'] == False) & (merged_df['sentiment_analysis'] == 0)]

    # Agrupar por desarrollador y contar recomendaciones negativas
    worst_devs = filtered_df['developer'].value_counts().head(3).index.tolist()

    return [{"Puesto {}".format(i+1): dev} for i, dev in enumerate(worst_devs)]

@app.get("/users-worst-developer/{año}")
async def users_worst_developer(año: int):
    return UsersWorstDeveloper(año)

def sentiment_analysis(empresa_desarrolladora: str):
    empresa_desarrolladora = empresa_desarrolladora.lower()  # Convertir a minúsculas el nombre del desarrollador ingresado
    merged_df = pd.merge(df_ur, df_sg, left_on='item_id', right_on='id')

    # Filtrar por desarrollador, convertir a minúsculas para la comparación
    filtered_df = merged_df[merged_df['developer'].str.lower() == empresa_desarrolladora]

    sentiment_counts = filtered_df['sentiment_analysis'].value_counts().to_dict()

    return {empresa_desarrolladora: {"Negative": sentiment_counts.get(0, 0), "Neutral": sentiment_counts.get(1, 0), "Positive": sentiment_counts.get(2, 0)}}

@app.get("/sentiment-analysis/{empresa_desarrolladora}")
async def get_sentiment_analysis(empresa_desarrolladora: str):
    return sentiment_analysis(empresa_desarrolladora)








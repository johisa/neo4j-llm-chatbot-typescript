//NOTE: this is only [movieId,embedding] - the movies are populated to DB when course is created
//NOTE: They have embedded the movie plot
LOAD CSV WITH HEADERS
FROM 'https://data.neo4j.com/llm-fundamentals/openai-embeddings.csv'
AS row
MATCH (m:Movie {movieId: row.movieId})
CALL db.create.setNodeVectorProperty(m, 'embedding', apoc.convert.fromJsonList(row.embedding))
RETURN count(*);
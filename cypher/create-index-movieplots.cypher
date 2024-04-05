//NOTE: only adds embedding to existing movies. The movies are populated to DB when course is created
CREATE VECTOR INDEX `moviePlots` IF NOT EXISTS
FOR (n: Movie) ON (n.embedding)
OPTIONS {indexConfig: {
`vector.dimensions`: 1536,
`vector.similarity_function`: 'cosine'
}};
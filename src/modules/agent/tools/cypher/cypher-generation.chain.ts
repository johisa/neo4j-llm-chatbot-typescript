import { BaseLanguageModel } from "langchain/base_language";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";


const promptText = `
You are a Neo4j Developer translating user questions into Cypher to answer questions
about movies and provide recommendations.
Convert the user's question into a Cypher statement based on the schema.

You must:
* Only use the nodes, relationships and properties mentioned in the schema.
* When required, \`IS NOT NULL\` to check for property existence, and not the exists() function.
* Use the \`elementId()\` function to return the unique identifier for a node or relationship as \`_id\`.
    For example:
    \`\`\`
    MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
    WHERE a.name = 'Emil Eifrem'
    RETURN m.title AS title, elementId(m) AS _id, a.role AS role
    \`\`\`
* Include extra information about the nodes that may help an LLM provide a more informative answer,
    for example the release date, rating or budget.
* For movies, use the tmdbId property to return a source URL.
    For example: \`'https://www.themoviedb.org/movie/'+ m.tmdbId AS source\`.
* For movie titles that begin with "The", move "the" to the end.
    For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".
* Limit the maximum number of results to 10.
* Respond with only a Cypher statement.  No preamble.


Example Question: What role did Tom Hanks play in Toy Story?
Example Cypher:
MATCH (a:Actor {{name: 'Tom Hanks'}})-[rel:ACTED_IN]->(m:Movie {{title: 'Toy Story'}})
RETURN a.name AS Actor, m.title AS Movie, elementId(m) AS _id, rel.role AS RoleInMovie

Schema:
{schema}

Question:
{question}`

// tag::function[]
export default async function initCypherGenerationChain(
  graph: Neo4jGraph,
  llm: BaseLanguageModel
) {
  const cypherPrompt = PromptTemplate.fromTemplate( promptText )

  return RunnableSequence.from<string, string>([
    {
      // new RunnablePassthrough() enables calling "chain.invoke('some question')" (onlye a string argument),
      // Otherwise an object would be required as argument since the prompt template is expecting
      // "question" and "schema" properties.
      // This is because in a RunnableSequence the output is passed to the next as input,
      // so the "question" and "schema" will be passed to the prompt.
      question: new RunnablePassthrough(),
      schema: () => graph.getSchema(),
    },
    cypherPrompt,
    llm,
    new StringOutputParser(),
  ]);
}
// end::function[]

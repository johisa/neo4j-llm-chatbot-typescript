import {
  Runnable,
  RunnablePassthrough,
  RunnablePick,
} from "@langchain/core/runnables";
import { Embeddings } from "@langchain/core/embeddings";
import initGenerateAnswerChain from "../chains/answer-generation.chain";
import { BaseLanguageModel } from "langchain/base_language";
import initVectorStore from "../vector.store";
import { saveHistory } from "../history";
import { DocumentInterface } from "@langchain/core/documents";
import { AgentToolInput } from "../agent.types";

// tag::throughput[]
type RetrievalChainThroughput = AgentToolInput & {
  context: string;
  output: string;
  ids: string[];
};
// end::throughput[]

// tag::extractDocumentIds[]
// Helper function to extract document IDs from Movie node metadata
const extractDocumentIds = (
  //NOTE: ([key: string]: any) === potentially any number of other properties of any type.
  //  But, we're only interested in the document's _id property
  documents: DocumentInterface<{ _id: string; [key: string]: any }>[]
): string[] => documents.map((document) => document.metadata._id);
// end::extractDocumentIds[]

// tag::docsToJson[]
// Convert documents to string to be included in the prompt
const docsToJson = (documents: DocumentInterface[]) =>
  JSON.stringify(documents);
// end::docsToJson[]

// tag::function[]
export default async function initVectorRetrievalChain(
  llm: BaseLanguageModel,
  embeddings: Embeddings
): Promise<Runnable<AgentToolInput, string>> {
  const vectorStore = await initVectorStore(embeddings);
  const vectorStoreRetriever = vectorStore.asRetriever(5);
  const answerChain = initGenerateAnswerChain(llm)

  // Build chain using super custom flow to link retrieved movies as context to the conversation history Response/Context.
  return (

    // CONTEXT retrieval step 1 - get documents using vector search
    //    Because the chain will receive an object as the input, you can use
    //    RunnablePassthrough.assign() to modify the input directly rather than the
    //    RunnableSequence.from() method used in the previous lessons.
    //    This should be used to collect relevant context using the retriever.
    RunnablePassthrough
      .assign({
        documents: new RunnablePick("rephrasedQuestion").pipe(
          vectorStoreRetriever
        ),
      })

    // CONTEXT retrieval step 2 - NOTE important to use the "documents" key from previous step!
    //  Next, the elementIds of the document must be extracted from the documents to create the
    //   :CONTEXT relationship between the (:Response) and (:Movie) nodes.
    //    At the same time, the context needs to be converted to a string, so it
    //    can be used in the Answer Generation Chain.
    //  The RunnablePassthrough is a fluent interface, so the .assign() method
    //   can be called to chain the steps together.
    //
      .assign({
        ids: new RunnablePick("documents").pipe(extractDocumentIds),
        //Convert documents to string
        context: new RunnablePick("documents").pipe(docsToJson),
      })

    //  The rephrased question and context can then be passed to the answerChain
    //   to generate an output.
    //
      .assign({
        output: (input: RetrievalChainThroughput) => {
          return answerChain.invoke({
            question: input.rephrasedQuestion,
            context: input.context,
          });
        }
      })

    //  Then, the input, rephrased question and output can be saved to the database
    //   using the saveHistory() function created in Conversation Memory module.
    //
      .assign({
        responseId: async (input: RetrievalChainThroughput, options) =>
          saveHistory(
            options?.config.configurable.sessionId,
            "vector",
            input.input,
            input.rephrasedQuestion,
            input.output,
            input.ids
          )
      })
    // Before, finally picking the output as a string.
      .pick("output")
  );
}
// end::function[]

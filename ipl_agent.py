from rag_chain import get_rag_chain


class IPLAgent:
    def __init__(self):
        self.chain = get_rag_chain()

    def query(self, question):
        result = self.chain.invoke({"query": question})
        return result

class JaccardSimilarity:
    def __init__(self, first_text, second_text):
        self.first_words = first_text.lower().split()
        self.second_words = second_text.lower().split()
        self.common_words = []
        self.incommon_words = []

    def calculate_similarity(self):
        for word in self.first_words:
            if word in self.second_words:
                self.common_words.append(word)

        for word in self.second_words:
            if word not in self.first_words:
                self.incommon_words.append(word)

    def calculate_probability(self):
        result = len(self.common_words) / (len(self.incommon_words) + len(self.first_words))
        return result

x = JaccardSimilarity("Machine Learning is a subset of AI", "Aim of using Machine Learning is to drive an AI application")
x.calculate_similarity()

if x.calculate_probability() > 0.5:
    print(f"The texts are similar. The Jaccard similarity probability is: {x.calculate_probability()}")
else:
    print(f"The texts are not similar. The Jaccard similarity probability is: {x.calculate_probability()}")

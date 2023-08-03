class Feedback:
    def __init__(self):
        self.user_feedback = {"plot_table": "NA", "summary": "NA", "query": "NA"}

    def plot_or_table_liked(self):
        self.user_feedback["plot_table"] = "liked"
        print(self.user_feedback)

    def plot_or_table_disliked(self):
        self.user_feedback["plot_table"] = "disliked"
        print(self.user_feedback)

    def summary_liked(self):
        self.user_feedback["summary"] = "liked"
        print(self.user_feedback)

    def summary_disliked(self):
        self.user_feedback["summary"] = "disliked"
        print(self.user_feedback)

    def query_liked(self):
        self.user_feedback["query"] = "liked"
        print(self.user_feedback)

    def query_disliked(self):
        self.user_feedback["query"] = "disliked"
        print(self.user_feedback)

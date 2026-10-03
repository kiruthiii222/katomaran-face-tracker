class VisitorCounter:
    """The unique visitor count is derived from registered identities in the database."""
    def __init__(self, database):
        self.database = database

    @property
    def unique_visitors(self):
        return self.database.unique_visitor_count()

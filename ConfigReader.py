# ConfigReader.py
class ConfigReader:
    def __init__(self, filename):
        self.filename = filename

    def read_lines_without_comments(self):
        """Read lines from the file, ignoring lines that start with '#'."""
        with open(self.filename, 'r') as file:
            lines = [line.strip() for line in file if not line.strip().startswith('#') and line.strip()]
        return lines

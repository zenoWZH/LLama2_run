# ConfigReader.py
class ConfigReader:
    def __init__(self, filename):
        self.filename = filename

    def read_lines_without_comments(self):
        """Read lines from the file, excluding lines that are comments or contain only whitespace."""
        lines = []
        with open(self.filename, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if self.is_valid_line(stripped_line):
                    lines.append(stripped_line)
        return lines

    def is_valid_line(self, line):
        """Check if a line is not a comment and not empty (ignores whitespace-only lines)."""
        return not line.startswith('#') and bool(line)


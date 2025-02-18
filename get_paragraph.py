import random

DEFAULT_BOOK_NAME = "war_and_peace.txt"


def get_book(title: str=DEFAULT_BOOK_NAME):
    with open(title, "r") as file:
        return file.read()


def trim_header(book: str, skip_first_chars: int=2000):
    return book[skip_first_chars:]


def get_chunk_from(book: str, chunk_size: int=1000):
    chunks = [book[i:i+chunk_size] for i in range(0, len(book), chunk_size)]
    return random.choice(chunks)


def get_random_paragraph_from(chunk: str):
    paragraphs = chunk.split("\n\n")
    return random.choice(paragraphs)


def get_paragraph_from_war_and_peace():
    book = get_book()
    book = trim_header(book)
    chunk = get_chunk_from(book)
    random_paragraph = get_random_paragraph_from(chunk)

    return random_paragraph
    

def main():
    get_paragraph_from_war_and_peace()


if __name__ == "__main__":
    main()


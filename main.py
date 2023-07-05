import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from embedding_search.community_map import download_datafile
from embedding_search.crossref import query_crossref
from embedding_search.academic_analytics import get_units, get_faculties, get_articles
from embedding_search.data_model import Article, Author
from tqdm import tqdm

load_dotenv()

AUTHORS_DIR = Path("authors/")
COMMUNITY_DF = download_datafile()


def list_downloaded_authors() -> list[str]:
    """List downloaded authors."""

    downloaded = AUTHORS_DIR.glob("*.json")
    return [author.stem for author in downloaded]


# def append_community_info(author: Author) -> Author:
#     """Add email address and community name to author."""

#     id_to_community_name = {
#         row.orcid: row.community_name for row in COMMUNITY_DF.itertuples()
#     }
#     id_to_email = {row.orcid: row.email for row in COMMUNITY_DF.itertuples()}
#     author.email = id_to_email[author.orcid]
#     author.community_name = id_to_community_name[author.orcid]
#     return author


def append_embeddings(author: Author) -> Author:
    """Get embeddings for authors' articles."""

    embeddings = OpenAIEmbeddings()
    # embed all articles in batch (m articles, n dimensions)
    author.articles_embeddings = embeddings.embed_documents(author.texts)
    return author


def parse_article(article: dict) -> Article:
    """Parse an article from the academic analytics API."""
    article = Article(
        doi=article["digitalObjectIdentifier"],
        title=article["title"],
        abstract=article["abstract"],
    )

    # Get cited by count from crossref
    if article.doi:
        crossref_data = query_crossref(article.doi, fields=["is-referenced-by-count"])
        if crossref_data["is-referenced-by-count"] is not None:
            article.cited_by = crossref_data["is-referenced-by-count"]

    return article


def download_one_author(author: dict) -> None:
    """Parse an author from the academic analytics API."""
    author = Author(
        id=author["id"],
        first_name=author["firstName"].title(),
        last_name=author["lastName"].title(),
    )

    articles = get_articles(author.id)
    author.articles = [parse_article(article) for article in tqdm(articles)]

    if author.articles:
        author = append_embeddings(author)
        author.save(AUTHORS_DIR / f"{author.id}.json")
    else:
        logging.info(f"Skipping {author.id} because they have no articles.")


def download_authors(overwrite: bool = False) -> None:
    """Download authors from ORCID and their papers from CrossRef."""

    downloaded = [] if overwrite else list_downloaded_authors()

    units = get_units()
    for unit in units:
        faculties = get_faculties(unit["unitId"])
        for faculty in faculties:
            if faculty["id"] in downloaded:
                continue
            try:
                download_one_author(faculty)
            except Exception as e:
                print(f"Error downloading {faculty['id']}: {e}")
                continue


def main() -> None:
    """Main function."""
    download_authors(overwrite=False)


if __name__ == "__main__":
    main()

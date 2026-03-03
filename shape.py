import json

FILE = "data/arxiv-metadata-oai-snapshot.json"
OUTPUT = "data/cleaned.jsonl"

# streaming loader (generator function)
def get_data(path):
    with open(path, "r", encoding="utf-8") as f:    # open file in read mode
        for line in f:                              # read every line
            yield json.loads(line)                  # yield is used to pause execution to make 
                                                    # function memory effieicent, as dataset is quite large 
                                                    # json.loads converts JSON into python dict
                                                    

# clean dataset
def clean_filter():
    total = 0
    kept = 0
    with open(OUTPUT, "w", encoding="utf-8") as f:
          for paper in get_data(FILE):      # get_data loads one paper at a time
              total+=1
              created = paper["versions"][0]["created"]     # using the first version's year as the year
              year = int(created.split()[-3])
              abstract = paper.get("abstract", "").strip()
              categories = paper.get("categories", "")
              if year >= 2019 and abstract and "cs." in categories: # if the year >= 2019, abstract exists, and cs category, keep paper
                  cleaned = {                               # cleaned only keeps certain fields
                      "id": paper.get("id", ","),
                      "title": paper.get("title", "").strip(),
                      "abstract": abstract,
                      "authors": paper.get("authors", ""),
                      "categories": paper.get("categories", ""),
                      "year": year
                  }
                  f.write(json.dumps(cleaned) + "\n")   # json.dumps converts dict into json and returns a string
                  kept+=1                               # need dumps to add newline
              if total % 100000 == 0:
                  print(f"Processed: {total}, Kept: {kept}")
              

                                               

if __name__ == "__main__":
    clean_filter()
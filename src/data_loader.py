def process_documents(files):
    documents = []
    total_size = 0
    total_words = 0
    doc_stats = []

    for file in files:
        content = file.read().decode("utf-8")
        documents.append(content)

        size = len(content)
        words = len(content.split())

        total_size += size
        total_words += words

        doc_stats.append({
            "name": file.name,
            "size": size,
            "words": words
        })

    avg_words = total_words / len(files) if files else 0

    return documents, total_size, total_words, avg_words, doc_stats

from korea_news_crawler.articlecrawler import ArticleCrawler

Crawler = ArticleCrawler()
Crawler.set_category('금융')
Crawler.set_date_range(2021, 7, 2021, 7)
Crawler.start()
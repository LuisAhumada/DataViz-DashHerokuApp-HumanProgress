


# A web app for the study and modeling of human progress

Data from various academic institutions and international organizations shows enormous improvements in human well-being throughout the world in the past centuries. Unfortunately, there seems to be a gap between reality, characterized by improvements, and public perception, which tends to be negative about the current state of the world, skeptical about the world's future, or illiterate about the causes of progress. Projects like HumanProgress.org gather empirical data from reliable sources that look at worldwide long-term trends and host it in an accessible and interactive online platform designed for students, researchers, and the general public. This project goes one step further by developing a new interactive tool to visualize the relationship between variables, causation through machine learning models such as linear regression, and prediction through neural networks. The project introduces an online dashboard powered by Plotly and delivers two case studies analyzing the variable life expectancy.

Plotly Dash webapp powered by Heroku, available at: https://humanprogressorg.herokuapp.com/

### Luis Ahumada Abrigo M.S.
#### The George Washington University

Repository includes:

- Scraping.py: Downloads all data from HumanProgress.org using BeautifulSoup
- Variables.py: Data preprocessing, data cleaning for all variables and countries data from HumanProgress.org 
- Model.py: Two models: Multiple Linear Regression and DNN Regression
- ModelDash.py: Model function applied to the data visualization WebApp.
- Viz (folder): Plotly Dash webapp powered by Heroku, available at: https://humanprogressorg.herokuapp.com/
- result.csv: Data for modeling
- df_all.csv: Data for the Web App
- DATS 6501 - Capstone Project - Luis Ahumada Abrigo.pdf: PDF Final Report




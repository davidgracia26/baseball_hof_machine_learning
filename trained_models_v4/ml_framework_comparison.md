## Problem:

Is there a way to predict based on statistics or awards or accolades if someone will make the Pro Baseball Hall of Fame?

## Analysis:

This project started as a ML.Net project but as you can see this quickly became a python project.
Initially, the goal of this project was to use a language I work with (C#) to learn ML priniciples quickly.
For the duration of the ML.Net portion of the project, it was tedious and it felt like a second class citizen in the .NET ecosystem. Running models involved creating pipelines and hoping that your data is transformed correctly. The main hindrance is that there is not a DataFrame in ML.Net so it just uses classes/records to handle the processing of the data. At one point cross validation was desired but it was simply not possible.

At that point, after struggling for a couple of months with it, I re-wrote the entire project in python. The re-write only took a few hours and after the re-centering of the project around python, which treats machine learning as a first-class citizen, the project took off. 

I tested several configurations of features optimizing for precision because I want the model to gatekeep the Hall of Fame. I started with hoping advanced metrics like OPS, K/9, WHIP and other stats that were not counting stats would be good indicators for prediction. 

Alas, they proved to produced a precision near 0.4. It was only when I started using counting stats that the precision topped 0.5. But, when I included only awards and accolades did the precision reach its highest that I found (0.8). The accuracy for this set of features also satisfied being greater than the rate of players that do not become Hall of Famers (98.7%) 

Currently the only model using is the DecisionTreeClassifier. There will be a venture into other models such as LightGBM and XGBoost to see if they can provide a higher precision.



Did the pipeline run cleanly on the first try? If not, what failed and how did you fix it?

The pipeline did not run cleanly first time around. I completely forgot to create the .env file. Second issue I had is the OPENAI_KEY variable was miss spelled. I had OPEN_API vs OPENAI_API.

After that, I had to go back and change the transformation stage. In the previous lesson, I used a break statement to stop the loop once 24 records had been processed. Because this time I had to use a function for the prefect pipeline, I switched over to a return statement. Other than that, it was just syntax lookup and doc reading. 


What did the Prefect UI show? Were there any retries?

After the bugs were fixed, it ran once with no problems. 

What is one thing you would change or add if you were deploying this pipeline to run on a daily schedule?

According to the prefect documentation, if doing it with CLI, I would need to use schedule create my-flow/my-deployment --interval 1800 -> 1800 being the number of seconds in 30 minutes.

Otherwise serve() with the schedule=Interval argument would work. That would require experimentation to fully grasp how the syntax works. 
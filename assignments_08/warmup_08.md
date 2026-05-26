
# ------------------------------------ Cloud Concepts --------------------------------


#Q1
 Reoccuring payment, often monthly, for shared infrastructure rather than buying/owning hardware. You don't have control over the server but there is greater redundancy which allows application uptime to be at 99%.



2. Vertical scaling is when you upgrade a single machine to meet compute demand. Horizontal scaling is when you spread the demand across different computers. 
    
    3 Scenarios questions:
        1. For a webapp that may see explosive growth, horizontal scaling. Vertical scaling sets you up for a single source of failiure. For a webapp, the result can mean an incredible loss of credibility and revenue. Horizontal scaling scatters demand across machines, making it more resilient. 
        2. For more GPU and Ram, vertical scaling. If the model training job requires previous work for new work to be produced, you are better off upgrading the current machine. 
        3. Horizontal scaling if the work is able to be split across machines. Having a single machine introduces a single point of failure risk. Having multiple machines mitigates that risk while being cheaper than a single machine.


#Q3 

Gmail - SaaS. It is a fully developed, ready to use software solution
Azure VM - PaaS. SWE's use it to develop software safely.
Azure App Service - PaaS. Same as Azure VM.
AWS S3 - LaaS. Online storage solution people use to store files.
GitHub Codespaces - PaaS - Online platform for developers to develop software
Snowflake - SaaS - Fully built solution that manages underlying infrastructure and applications out of the box.

laaS = Infrastructure as a service. You rent hardware/existing infrastructure like servers for use. When looking to store files online, laaS. 

PaaS = Platform as a Service. Offers a framework and environment developers can use. Google App Engine or Heroku are examples of PaaS services.

SaaS - Ready built software hosted online that you can access. Usually accessed with a web brower. Examples are: Youtube, Slack, Rockwell Automation Plex ERP

Developer Responsibilities (in order):

laaS - Manage operating systems, applications and network security. 
PaaS - code and data. Platform maintenance handled by provider.
SaaS - User access, data governance, application configuration.
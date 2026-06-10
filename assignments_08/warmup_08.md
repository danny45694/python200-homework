
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
Azure VM - LaaS. SWE's use it to develop software safely.
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

Cloud Concepts Questions

Q4. Snowflake offers a specialized, managed for you service. It handles the underlying structure. Azure is provides you with a generalized cloud environment. Because of those differences, Snowflake is only used for Data Warehousing, engineering and sharing. Azure allows much more flexibility in the applications hosted in exchange for having to manage setup and maintenance yourself.

Q5. If dataset fits on a single machine and you don't have massive compute demands, local processing is better. The other situation where local is better is when handling highly sensitive, regulated data.

Azure Basics

Q1. Azure subscription is the billing account that owns all the resources in an organization. Resource group is a sandbox that bundles all your related cloud resources together. Resource group lies within the subscription.

Q2. Ephemeral - Each time you close shell, everything gets deleted. To ensure it sticks around, Cloud Shell needs to be connected to file share. File share is a named storage folder in Azure.

Q3. SSH operates in key pairs. They prove identity without transmitting a password. Private key stay on local machines while public key is uploaded to the systems you want access to. When connecting SSH verifies the key match, bypassing the need for a password crossing the network.

Q4. --output table displays only high-level properties such as Name, CloudName, SubscriptionId, and State. 

daniel [ ~/clouddrive ]$ az account show
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#danieladiazop@gmail.com",
    "type": "user"
  }
}



# laravel_weather_prediction
This laravel project uses linear regression to predict the maximum temperature for a given place.  
It uses the MVC pattern for inserting data into a sqlite database and for returning views.  
A python script is used for making API calls to https://open-meteo.com/ to collect data, which is then fitted into a linear regression model.  

# Usage
Please make sure that you have laravel installed, as well as python3 with all the needed packages.  
Changes to .env are also necessary to make it work on your machine.  

```bash
php artisan serve
```

Tested with Python 3.11.5 

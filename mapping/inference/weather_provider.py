from pathlib import Path
import pandas as pd

class WeatherProvider: 

    def __init__(
        self, 
        weather_file = "data/interim/master_dataset.parquet"
    ):
        
        self.weather_file = Path(weather_file)
        self.weather = self._load_weather()

    def get_weather_history(
        self, 
        target_date,
        mountain, 
        history_days = 60
    ):
        
        target_date = pd.Timestamp(target_date)

        if mountain not in self.weather["mountain"].unique():
            
            raise ValueError(
                f"Mountain '{mountain}' is not included in the model."
            )
        
        mountain_weather = self.weather[
            self.weather["mountain"] == mountain 
        ].sort_values("date")

        if target_date not in mountain_weather["date"].values: 
            
            raise ValueError(
                f"No weather data for {mountain} on {target_date.date()}."
            )
        start_date = target_date - pd.Timedelta(days=history_days)

        history = mountain_weather[
            (mountain_weather["date"] >= start_date)
            & (mountain_weather["date"] <= target_date)
        ].copy()

        return history
    
    def get_forecast_day(
        self, 
        target_date, 
        mountain
    ): 
        
        target_date = pd.Timestamp(target_date)

        if mountain not in self.weather["mountain"].unique():
            
            raise ValueError(
                f"Mountain '{mountain}' is not included in the model."
            )
        
        forecast = self.weather[
            (self.weather["mountain"] == mountain)
            & (self.weather["date"] == target_date)
        ].copy()

        if forecast.empty:
            raise ValueError(
                f"No weather data for {mountain} on {target_date.date()}."
            )

        return forecast
    
    def get_weather(
        self,
        mountain,
        date,
    ):
        return self.get_forecast_day(
            target_date=date,
            mountain=mountain,
        )
    
    def _load_weather(self):

        weather = pd.read_parquet(self.weather_file)

        weather["date"] = pd.to_datetime(weather["date"])

        return weather




        

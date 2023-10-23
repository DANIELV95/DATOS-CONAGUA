## Example of the use of drawre function
library(IETD)
library(ggplot2)
library(scales)
require(reticulate)
path_to_python <- "C:/Users/HIDRAULICA/anaconda3/envs/hidraulica/python.exe"
use_python(path_to_python)
pd <- import("pandas")

# reticulate::install_miniconda(force=TRUE, update=FALSE)

files = list.files('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS/', pattern = "\\.csv$")

for(file in files){
  print(file)
  # file = files[2]
  file_path = gsub(" % ","",paste('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS/','%',file))
  folder_name = gsub(" % ","",paste(substr(file,1,nchar(file)-8),'%','_Events'))
  folder_path = gsub(" % ","",paste('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS_Events/','%',folder_name,'%','/'))
  
  folder_exists = dir.exists(folder_path)
  if(!folder_exists){
    dir.create(folder_path)
  }

  # classes <- c('POSIXct', 'factor', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL')
  ema <- read.csv(file = file_path) #, colClasses = classes)
  ema <- ema[c('X', 'P')]
  ema$X <- as.POSIXlt(ema$X, tz='UTC', format= "%Y-%m-%d %H:%M:%OS")
  ema$P <- as.numeric(ema$P)
  colnames(ema) <- c('t', 'P')
  df <- ema

  df <- r_to_py(df) #Transform to Pandas DataFrame
  df <- df$set_index(pd$DatetimeIndex(df['t']))
  #df_meidan_hours=df$resample('1H', how='median', closed='left', label='left')
  dfsum <- df$resample('1H')$agg('sum')
  dfsum <- py_to_r(dfsum) #Transform back to r's data.frame
  
  t = as.POSIXct(rownames(dfsum), tz='UTC')
  P = dfsum$P
  df = data.frame(t=t, P=P)
  
  ## load a time series (an artificial data that is included in the package) and plot it.
  # ggplot(df,aes(t, P)) + 
  #   theme_bw()+
  #   geom_bar(stat = 'identity',colour="black",lwd=1, fill="gray")+
  #   scale_x_datetime(labels = date_format("%Y-%m-%d %H"))+
  #   xlab("Date")+
  #   ylab("Rainfall depth [mm]")
  
  IETD_value = 6
  Thres_value = 50
  
  # events <- drawre(df,IETD=IETD_value,Thres=Thres_value)$Rainfall_Events
  events <- try(drawre(df,IETD=IETD_value,Thres=Thres_value)$Rainfall_Events, TRUE)
  # events[1]
  if (!inherits(events, "try-error")) {
  
    # write.csv(events[[1]], file = 'D:/DANI/2023/TEMA_TORMENTAS/DATOS/MONTERREY_Events.csv')
    
    ema_event = substr(folder_name,1,nchar(folder_name)-1)
    for(i in 1:length(events)) {
      file_name = gsub(" % ","",paste(folder_path,'%',ema_event,'%','_','%',sprintf('%02d',i),'%','.csv'))
      write.csv(events[[i]], file = file_name, row.names = FALSE)
      # print(i)
    }
    
    print(length(events))
    file_events<-gsub(" % ","",paste('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/EventsTXT/','%',ema_event,'%','s_IETD','%',IETD_value,'%','_Thres','%',Thres_value,'%','.txt'))
    sink(file_events)
    print(events)
    sink()
  } else {
    unlink(folder_path, recursive=TRUE)
  }
}
  


  # Event13<-events[[13]]
  # 
  # events[]
  # events[[1]]$t
  # 
  # ggplot(Event13,aes(x=t,y=P)) +
  #   theme_bw()+
  #   geom_bar(stat = 'identity',colour="black", lwd=1, fill="gray")+
  #   scale_x_datetime(labels = date_format("%Y-%m-%d %H"))+
  #   ylab("Rainfall depth [mm]")
  # 
  # Results_CVA<-CVA(Time_series=df,MaxIETD=24)
  # IETD_CVA<-Results_CVA$EITD
  # Results_CVA$Figure
  # 


# colnames(Time_series) <- c('t', 'P')
# 
# Rainfall.Eevents<-drawre(Time_series,IETD=5,Thres=20)$Rainfall_Events
# Rainfall.Eevents
# 
# Time_series<-hourly_time_series
# ggplot(Time_series,aes(x=Date,y=Rainfall.depth)) +
#   theme_bw()+
#   geom_bar(stat = 'identity',colour="black",lwd=1, fill="gray")+
#   scale_x_datetime(labels = date_format("%Y-%m-%d %H"))+
#   ylab("Rainfall depth [mm]")
# 
# ## Apply drawre function to Time_series to extract independent rainfall events by considering IETD=5 and a rainfall depth threshold of 0.5 to define slight rainfall events.
# Rainfall.Eevents<-drawre(Time_series,IETD=5,Thres=0.5)$Rainfall_Events 
# 
# ## Plot the extracted event # 13
# Event13<-Rainfall.Eevents[[13]]
# ggplot(Event13,aes(x=Date,y=Rainfall.depth)) + 
#   theme_bw()+
#   geom_bar(stat = 'identity',colour="black", lwd=1, fill="gray")+
#   scale_x_datetime(labels = date_format("%Y-%m-%d %H"))+
#   ylab("Rainfall depth [mm]")
# 
# 
# Results_CVA<-CVA(Time_series=hourly_time_series,MaxIETD=24)
# IETD_CVA<-Results_CVA$EITD
# Results_CVA$Figure

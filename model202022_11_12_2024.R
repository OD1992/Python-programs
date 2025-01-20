### R script for analysis of Senegal dataset (type Senegal model)
rm(list=ls())
gc()

#load libraries


library(sf)
library(snakecase)
library(lubridate)
library(janitor)
library(tidyverse)
library(viridis)
library(RColorBrewer)

pkg<-c("raster","INLA","TMB","sparseMVN","Matrix","rgdal","gdistance",
       "malariaAtlas","sp","rgeos","foreach","doParallel")
for (i in 1: length(pkg)){library(pkg[[i]],character.only = TRUE)}

setwd(dir="/mnt/efs/stratification/CountryFolders/Senegal/")
#adapt paths below#########################
#dirpath <- "/home/odiao/"
dirpath <- "Projects/1.0 Risk Map/2022/"
outputpath <- "/home/odiao/Output_11_20_2024/"



# Step 1: Read in new case data ======
coordinates <- read_csv("/mnt/efs/stratification/CountryFolders/Senegal/Projects/1.0 Risk Map/2022/01 Raw data/Senegal data for MAP/Org Unit - Corrected Coordinates2.csv") %>% 
  dplyr::select(adm1 = `Org Unit Name Level 2`, adm2 = `Org Unit Name Level 3`,
                hf = `Org Unit Short Name`, lat = Latitude, long = Longitude) %>% 
  mutate(adm1 = snakecase::to_snake_case(adm1),
         adm2 = snakecase::to_snake_case(adm2),
         hf = snakecase::to_snake_case(hf),
         unique_id = paste("sen",row_number(), sep="_"))

data_2020 <- readxl::read_xlsx(paste0(dirpath, "01 Raw data/PNLPdata/Extraction Data Paludisme DS FS DHIS2 2020_2021_2022.xlsx"), sheet = 1)
data_2021 <- readxl::read_xlsx(paste0(dirpath, "01 Raw data/PNLPdata/Extraction Data Paludisme DS FS DHIS2 2020_2021_2022.xlsx"), sheet = 2)
data_2022 <- readxl::read_xlsx(paste0(dirpath, "01 Raw data/PNLPdata/Extraction Data Paludisme DS FS DHIS2 2020_2021_2022.xlsx"), sheet = 3)


data_2020_2022 <- bind_rows(data_2020, data_2021, data_2022) %>% 
  janitor::clean_names() %>% 
  separate(mois, into = c('monthname', 'year'), sep = " ") %>% 
  mutate(month = factor(monthname, levels = c("Janvier","Février","Mars","Avril","Mai","Juin",
                                              "Juillet","Août","Septembre","Octobre","Novembre","Décembre")),
         month = as.numeric(month),
         year = as.numeric(year),
         year.month= paste(year, month, sep = "."),
  ) %>% 
  mutate(adm1 = snakecase::to_snake_case(regions),
         adm2 = snakecase::to_snake_case(districts),
         hf = snakecase::to_snake_case(formation_sanitaire),
         conf_u5 = cas_confirmes_feminin_0_4_ans + cas_confirmes_masculin_0_4_ans,
         conf_ov5 = cas_confirmes_feminin_5_ans_excluant_les_femmes_enceintes + cas_confirmes_masculin_5_ans_excluant_les_femmes_enceintes,
         pres_all = NA,
         pres_u5 = NA,
         pres_ov5 = NA,
         pres_preg = NA) %>% 
  dplyr::select(adm1 = adm1, adm2 = adm2, hf, month, year , year.month, 
                allout_all = total_consultants_toutes_affections_confondues,
                allout_ov5 = consultants_toutes_affections_confondues_5_ans_excluant_les_femmes_enceintes,
                allout_preg = consultants_toutes_affections_confondues_femme_enceinte,
                allout_u5 = consultants_toutes_affections_confondues_0_4_ans,
                conf_all = total_cas_confirmes, 
                conf_ov5,
                conf_preg = cas_confirmes_femme_enceinte,
                conf_u5,
                pres_all, pres_ov5, pres_preg, pres_u5,
                test_all = total_tests_tdr_realises,
                test_ov5 = tests_tdr_realises_5_ans_excluant_les_femmes_enceintes,
                test_preg = tests_tdr_realises_femme_enceinte,
                test_u5 = tests_tdr_realises_0_4_ans,
                conf_f_all = total_cas_confirmes_feminin,
                conf_f_ov5 = cas_confirmes_feminin_5_ans_excluant_les_femmes_enceintes,
                conf_f_u5 = cas_confirmes_feminin_0_4_ans,
                conf_m_all = total_cas_confirmes_masculin,
                conf_m_ov5 = cas_confirmes_masculin_5_ans_excluant_les_femmes_enceintes,
                conf_m_u5 = cas_confirmes_masculin_0_4_ans,
                susp_all = total_cas_suspect,
                susp_ov5 = cas_suspect_5_ans_excluant_les_femmes_enceintes,
                susp_preg = cas_suspect_femme_enceinte,
                susp_u5 = cas_suspect_0_4_ans) %>% 
  left_join(coordinates)

unique_points <- data_2020_2022 %>% 
  distinct(adm1, adm2, unique_id, hf, lat, long) %>% 
  filter(!is.na(lat)&!is.na(unique_id)) %>% 
  #mutate(id = paste("sen",row_number(), sep="_")) %>% 
  dplyr::select(unique_id = unique_id, hf, x = long, y = lat) 

#setwd("Z:/Stratification/Country_folders/Senegal/Projects/1.0 Risk Map/04 Models/monthly")
#***** Step 1: Read in new case data
#casespath<-"Z:/Stratification/Country_folders/Senegal/Projects/1.0 Risk Map/02 Cleaned data/"
#outputpath<-"Z:/Stratification/Country_folders/Senegal/Projects/1.0 Risk map/05 Outputs/monthly/unmasked/"
#covpath <- "Z:/Stratification/Country_folders/Senegal/Common files/Covariates/cropped/"
#tspath <-"Z:/Stratification/Country_folders/Senegal/Projects/1.0 Risk map/03 Additional files (optional)/Population adjustments/"
#shpath<-"Z:/master_geometries/Admin_Units/Global/MAP/2019/"
# thfpath<-"Z:/Stratification/Country_folders/Senegal/Projects/1.0 Risk map/03 Additional files (optional)/TimeToHF/"
# setwd("C:/Users/andrep/Dropbox/Senegal/Models")
# casespath<-"C:/Users/andrep/Dropbox/Senegal/02 Cleaned data/"
# outputpath<-"C:/Users/andrep/Dropbox/Senegal/05 Outputs/annual/unmasked/"
covpath <- "Common files/Covariates/1km/"
#covpath_Monthly <- "Common files/Covariates/1km/Monthly/"

# tspath <-"C:/Users/andrep/Dropbox/Senegal/03 Additional files (optional)/Population adjustments/"
# shpath<-"C:/Users/andrep/Dropbox/Senegal/Common files/Shapefiles/"
#thfpath<-"C:/Users/andrep/Dropbox/Senegal/03 Additional files (optional)/TimeToHF/"
###########################################

####################Define months to be added in the suffix of saved objects################################
month <- c("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")
year <- 2020:2022
##########################################################################################################

#read.csv(paste0(casespath,"facility_cases",".csv"))
case.data.orig <- data_2020_2022 %>% 
  left_join(dplyr::select(unique_points, unique_id, hf)) %>% 
  dplyr::select(Org.Unit.ID = unique_id, Confirmed_cases_by_RDT = conf_all,year,month,
                long, lat) %>%
  filter(!is.na(lat))
#count cases used for the analysis, counts used with different column, and max potential counts
histdf<-aggregate(case.data.orig$Confirmed_cases_by_RDT, by=list(monthcases=case.data.orig$month), FUN=sum)
plot.new()
png(file=paste0(outputpath,"RDTcasespermonth.png"))
plot(histdf$monthcases,histdf$x,type="b")
dev.off();dev.off()


# Step 2: Read in population data and construct reference image
population <- raster("Common files/Population/SEN_population_v1_0_gridded_1km.tif")
# maybe need to replace NA population by 0 to make it work - this has to be checked!
NAvalue(population) <- 0
plot(population)

# Step 2b: make treatment seeking map (computed separately from travel time to nearest HF)
admin2 <- st_read("Common files/Shapefiles/Shapes files 79 Districts VF/Shapes files 79 Districts VF/districtsanitaire_79_clean.shp")
#admin2 <- st_read(paste0(dirpath, "districtsanitaire_79_clean.shp"))
admin0 <- st_union(admin2) %>% as_Spatial()
plot(admin0)
SEN<-admin0

#treatment seeking probability made in the code treat.seek.R
#***********************First version ***************************************
TS <- raster("Projects/1.0 Risk Map/2022/03 Additional files (optional)/Treatment seeking/ts_senegal_mean0_2023-03-29.tif")%>% mask(admin0)
#TS <- raster(paste0(dirpath, "ts_senegal_mean0_2023-03-29.tif"))
#check extent etc
TS;population
plot(TS,main="Treatment seeking")

#check extent etc

#verifies whether points fall in a given polygon.
###########################################CAUTION!!!!!!!!!!!!!!!!!!###############################################
######################THE MAIN POLYGON IN SENEGAL ADMIN 1 IS POLYGON NB 6 SO ONLY THIS ONE IS USED#################
in.country <- point.in.polygon(coordinates(population)[,1],coordinates(population)[,2],SEN@polygons[[1]]@Polygons[[1]]@coords[,1],SEN@polygons[[1]]@Polygons[[1]]@coords[,2])

###########################################CAUTION!!!!!!!!!!!!!!!!!!###############################################
######################THE MAIN POLYGON IN SENEGAL ADMIN 1 IS POLYGON NB 6 SO ONLY THIS ONE IS USED#################

#in.country <- which(in.country==1 & getValues(population)>0)# do not make prediction if population not above 0

in.country <- which(in.country==1)# we prefer this option since very low values from Worldpop not confirmed by satellite images
bigN <- length(in.country)
reference.image <- population
values(reference.image) <- NA
population <- getValues(population)[in.country]
#print(sum(!is.na(population))) gives 70704 values corresponding to the masked pop
population[is.na(population)] <- 0 # replace NA by 0

#same for TS
TS <- getValues(TS)[in.country]
TS[is.na(TS)] <- 0 # replace NA by 0
in.country.coords <- coordinates(reference.image)[in.country,]

plot(raster(paste0(covpath, "Aridity_Index_v2.Synoptic.Overall.Data.1km.Data.tif"),
            main="AI")%>% mask(admin0))

#####COMMENT LINES IN BOX IF WITHOUT PARALLELIZATION OF R######## 
# cores=detectCores()                                           #
# cl <- makeCluster(cores[1]-20) #not to overload your computer #
# registerDoParallel(cl)                                        #
# foreach(kk=1:length(month),.packages = pkg) %dopar% {         #
#################################################################
#old loop  
for(pp in 1:length(year)){
  for(kk in 1:length(month)){
    #for testing only
    #kk=1    
    case.data.orig <- data_2020_2022 %>% 
      left_join(dplyr::select(unique_points, unique_id, hf)) %>% 
      dplyr::select(Org.Unit.ID = unique_id, Confirmed_cases_by_RDT = conf_all, year,month,
                    long, lat) %>%
      filter(year == year[pp]) %>%
      filter(!is.na(lat))
    case.data<-case.data.orig[c("Org.Unit.ID","Confirmed_cases_by_RDT","month","long","lat")]
    case.data <- case.data[case.data$month==kk,]
    #caseag<-aggregate(case.data$Confirmed_cases_by_RDT, by=list(month=case.data$month,Org.Unit.ID=case.data$Org.Unit.ID),
    # FUN=sum, na.rm=TRUE)
    #names(caseag)<-c("month","Org.Unit.ID","cases")
    #test<-aggregate(caseag$cases, by=list(Org.Unit.ID=caseag$Org.Unit.ID),
    # FUN=mean, na.rm=TRUE)
    #test2 <- case.data[c("Org.Unit.ID","long","lat")]
    #test2 <- unique(test2)
    #case.data <- merge(test,test2,by="Org.Unit.ID")
    #names(case.data) <- c(c("Org.Unit.ID","total_cases","Long","Lat"))
    case.data<-case.data[c("Org.Unit.ID","Confirmed_cases_by_RDT","long","lat")]
    names(case.data) <- c(c("Org.Unit.ID","total_cases","Long","Lat"))
    case.data$total_Pf<-case.data$total_cases # we artificially add total_Pf only to avoid changing names in the model
    case.data <-droplevels(case.data)
    valid.fac <- which(!is.na(case.data$Lat) & !is.na(case.data$Long))
    Nfacilities <- length(valid.fac)
    
    
    # Step 3: Build covariate matrix
    covariate.names <- c("accessibility_to_cities_2015_v1.0",
                         "Aridity_Index_v2.Synoptic.Overall.Data.1km.Data",
                         "distance_to_water",
                         "MERIT_Elevation.Synoptic.Overall.Data.1km.mean",
                         "IGBP_Landcover_Class.09_Savannas.2020.Annual.Data.1km.fraction",
                         "PET_v2.Synoptic.Overall.Data.1km.Data",
                         "Global_Hybrid_Pop_v2_1km_UNAdj_2020",
                         "SRTM_SlopePCT_Corrected.Synoptic.Overall.Data.1km.Data",
                         "TSI.Martens2.Pf.2016.Annual.Mean.1km.Data",
                         "TWI",
                         "VIIRS.DNB.v2.1_Clean.Background.2022.Annual.Data.1km.mean",
                         paste0("chirps.v2.0.",paste(year[pp], month[kk], sep = "."),".sum.1km.NN"),
                         paste0("EVI_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data"),#
                         paste0("LST_Day_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data"),
                         paste0("LST_Night_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data"),
                         paste0("TCB_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data"),
                         paste0("TCW_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data"))
    
    #which type of transformation for which type of covariate?
    transformation.types <- c("Exponential",
                              "Normal",
                              "Exponential",
                              "Exponential",
                              "None",
                              "Normal",
                              "Exponential",
                              "Exponential",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal",
                              "Normal"
                              # "Normal"#,
                              #  "None"#for binary
    )
    Ncovariates <- length(covariate.names)
    for (i in 1:Ncovariates) {cat(covariate.names[i],"\t\t",transformation.types[i],col="\n")}
    
    covariates <- list()
    
    for (k in 1:Ncovariates) {
      cat("Processing covariate:",covariate.names[k],"...\n")
      #cov.current <- raster(paste(covpath,covariate.names[k],".tif",sep=""))
      if (covariate.names[k]==paste0("chirps.v2.0.",paste(year[pp], month[kk], sep = "."),".sum.1km.NN")|| covariate.names[k]==paste0("EVI_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data") || covariate.names[k]== paste0("LST_Day_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data")|| covariate.names[k]==paste0("LST_Night_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data")||covariate.names[k]==paste0("TCB_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data")||covariate.names[k]==paste0("TCW_v6.",paste(year[pp], month[kk], sep = "."),".mean.1km.Data")){
        cov.current <- raster(paste(covpath,"Monthly/",covariate.names[k],".tif",sep="")) }
      else{cov.current <- raster(paste(covpath,covariate.names[k],".tif",sep=""))}
      cov.current <-resample(cov.current, reference.image, method = "bilinear")  # Use "ngb" for categorical data
      if (!prod(dim(cov.current)==dim(reference.image))) {stop("Mismatched dimensions!\n")}
      cov.current <- getValues(cov.current)[in.country]
      if (transformation.types[k]=="Normal") {
        cov.current <- (cov.current-mean(cov.current,na.rm=TRUE))/sd(cov.current,na.rm=TRUE)
      }
      if (transformation.types[k]=="Exponential") {
        cov.current <- cov.current+min(cov.current[cov.current>0 & cov.current!=Inf & cov.current!=-Inf & !is.nan(cov.current)],na.rm=TRUE)+abs(min(cov.current[cov.current>0 & cov.current!=Inf & cov.current!=-Inf & !is.nan(cov.current)],na.rm=TRUE))
        cov.current <- qnorm(pexp(cov.current,1/mean(cov.current,na.rm=TRUE)))
      }
      nas <- which(is.na(cov.current))
      cat('NAs = ',length(nas),'\n')
      if (length(nas)>0) {
        validp <- which(!(is.na(cov.current) | cov.current==-Inf | cov.current==Inf))
        valid.coords <- in.country.coords[validp,]
        for (i in nas) {
          nearestv <- which.min((valid.coords[,1]-in.country.coords[i,1])^2+(valid.coords[,2]-in.country.coords[i,2])^2)
          cov.current[i] <- cov.current[validp[nearestv]]
        }
      }
      infs <- which(cov.current==Inf)
      cat('Infs = ',length(infs),'\n')
      if (length(infs)>0) {
        validp <- which(!(cov.current==-Inf | cov.current==Inf))
        valid.coords <- in.country.coords[validp,]
        for (i in infs) {
          nearestv <- which.min((valid.coords[,1]-in.country.coords[i,1])^2+(valid.coords[,2]-in.country.coords[i,2])^2)
          cov.current[i] <- cov.current[validp[nearestv]]
        }
      }
      neginfs <- which(cov.current==-Inf)
      cat('-Infs = ',length(infs),'\n')
      if (length(neginfs)>0) {
        validp <- which(!(cov.current==-Inf))
        valid.coords <- in.country.coords[validp,]
        for (i in neginfs) {
          nearestv <- which.min((valid.coords[,1]-in.country.coords[i,1])^2+(valid.coords[,2]-in.country.coords[i,2])^2)
          cov.current[i] <- cov.current[validp[nearestv]]
        }
      }
      hist(cov.current, main=covariate.names[k])
      cat("range: ",range(cov.current),"\n")
      Sys.sleep(3)
      covariates[[length(covariates)+1]] <- cov.current
    }
    covariates <- do.call(rbind,covariates)
    covariates[covariates > 3] <- 3 #truncate to mimic non-linearity in the extreme
    covariates[covariates < -3] <- -3 #truncate to mimic non-linearity in the extreme
    
    
    hf.ids <- unique(case.data$Org.Unit.ID)
    hf.longlats <- cbind(case.data$Long,case.data$Lat)[!duplicated(case.data$Org.Unit.ID),]
    
    ## Step 3: Read in traveltime surfaces and construct raw catchment populations
    
    #file.exists(paste0(dirpath, "ttdist",year[kk],".Rdata"))==TRUE)
    if(file.exists(paste0(dirpath, "ttdist",paste(year[pp], month[kk], sep = "."),".Rdata"))==TRUE){
      load(paste0(dirpath, "ttdist",paste(year[pp], month[kk], sep = "."),".Rdata"))
    }else{
      traveltime.distance.matrix <- matrix(0,nrow=bigN,ncol=Nfacilities)
      for (i in 1:Nfacilities) {
        #check the file path time to HFs here matches the reference.image surface you've used
        #*************************First version *****************************
        #eval(parse(text=paste("traveltime.distance.matrix[,",i,"] <- raster(\"/mnt/Z/stratification/CountryFolders/Senegal/Projects/1.0 Risk map/2022/03 Additional files (optional)/Treatment seeking/Time_to_HF/2023-03-29/",case.data$Org.Unit.ID[valid.fac[i]],"HF.access.tif\")[in.country]",sep="")))
        eval(parse(text=paste("traveltime.distance.matrix[,",i,"] <- raster(\"Projects/1.0 Risk Map/2022/03 Additional files (optional)/Treatment seeking/Time_to_HF/2023-03-29/",case.data$Org.Unit.ID[valid.fac[i]],"HF.access.tif\")[in.country]",sep="")))
        
        #convulated may to build up matrix col (HF) and row (pixels), eval remove
        reference.image[in.country] <- traveltime.distance.matrix[,i]
        image(reference.image)
        cat(i,"\n")
      }
      save(traveltime.distance.matrix, file = paste0(outputpath, "ttdist",paste(year[pp], month[kk], sep = "."),".Rdata"))#save as Rdata
    }
    
    traveltime.distance.matrix[traveltime.distance.matrix<1] <- 1
    invdistance <- 1/(traveltime.distance.matrix)
    catchments <- invdistance^2
    catchments <- catchments/rowSums(catchments+0.000000001)# normalise things .Why? sometimes you have no HF accessible
    
    for (i in 1:bigN) {
      catchment.ordering <- sort.list(traveltime.distance.matrix[i,],decreasing=F)
      n.less.than.20 <- length(which(traveltime.distance.matrix[i,]<20)) # if several HF within 20mn, then HF are counted
      catchment.list <- catchment.ordering[1:(max(n.less.than.20,1)+4)] # supposes attendance at one of nearest 5 HFs (counting those within a 20 min travel time as a single facility)
      invdistance[i,!(1:Nfacilities %in% catchment.list)] <- 0 # sparsity (add zeroes in the matrix when too far)
    }
    
    invdistance[traveltime.distance.matrix>180] <- 0# threshold people not allowed to travel up to 180mn
    
    catchments <- invdistance^2
    catchments <- catchments/rowSums(catchments+0.000000001)
    #catchment matrix is a matrix of : n facility by m pixels
    #rows or columns (to be checked): proportion of people going to facility i
    #value of matrix elements is: proportion of people who attend facility i
    
    invdists <- pixel.ids <- hf.ids <- numeric()
    for (i in 1:bigN) {
      nonzero <- which(invdistance[i,]>0)#for pixel i, which HF would people could go (prob>0)
      invdists <- c(invdists,invdistance[i,nonzero])# get distance to these selected HF
      pixel.ids <- c(pixel.ids,rep(i,length(nonzero)))# pixel id (indexing vector by pixel)
      hf.ids <- c(hf.ids,nonzero)# save the HF insto a vector
    }
    save(bigN,covariates,Nfacilities,population,TS,case.data,valid.fac,invdists,
         hf.ids,pixel.ids, file=paste0(outputpath,"prefit_total",paste(year[pp], month[kk], sep = "."),"unmasked.Rdata"))#to be checked maybe better Rdata
    rm(traveltime.distance.matrix)#release some RAM
  }#temporary end loop  
}
#stopCluster(cl) 


#old loop  
for (pp in 1:length(year)) {
  for(kk in 1:length(month)){
    
    load(paste0(outputpath,"prefit_total",paste(year[pp], month[kk], sep = "."),"unmasked.Rdata"))
    population[is.na(population)] <- 0 # replace NA by 0
    TS[is.na(TS)] <- 0 # replace NA by 0
    # Step 5: Build INLA mesh
    Senegal.mesh <- inla.mesh.2d(boundary=admin0@polygons[[1]],max.edge=c(0.1,2),cut=0.1)
    plot(Senegal.mesh);Senegal.mesh$n
    Senegal.spde <- (inla.spde2.matern(Senegal.mesh,alpha=2)$param.inla)[c("M0","M1","M2")]
    Senegal.A <- inla.spde.make.A(Senegal.mesh,in.country.coords)#inla.mesh.project(Senegal.mesh,in.country.coords)$A
    
    #For Win machine
    #Delete the file.o before calling compile()
    #compile("modelmonth.cpp") #
    dyn.load(dynlib("/home/odiao/modelmonth"))
    #
    input.data <- list(
      'bigN'=bigN,
      'static_covariate_matrix'=t(covariates),
      'spde'=Senegal.spde,
      'A'=Senegal.A,
      'nHFs'=Nfacilities,
      'population'=population,
      'treatment'=TS,
      ###################################check the year#########################################
      'HFcases_Pf'=case.data$total_Pf[valid.fac],# here just add the total annual
      #here we used total cases instead of Pv
      'HFcases_Pv'=case.data$total_cases[valid.fac],# here just add the total annual
      ###################################end check the year#####################################
      'Nunwrapped'=length(invdists),
      'invdists'=invdists,
      'hf_ids'=hf.ids,
      'pixel_ids'=pixel.ids
    )
    
    parameters <- list(
      'intercept_Pf'=0,
      'intercept_Pv'=0,
      'static_slopes_Pf'=rep(0,dim(covariates)[1]),
      'static_slopes_Pv'=rep(0,dim(covariates)[1]),
      'log_range'=-2,
      'log_sd'=1,
      'field_Pf'=numeric(Senegal.mesh$n),
      'field_Pv'=numeric(Senegal.mesh$n),
      'log_overdispersion_scale'=0
    )
    
    obj <- MakeADFun(input.data,parameters,DLL="modelmonth",random=c('field_Pf','field_Pv'))
    obj$fn()
    opt <- nlminb(obj$par,obj$fn,obj$gr,control=list(iter.max=300,eval.max=300))
    rep <- sdreport(obj,getJointPrecision = TRUE)
    save.image(file=paste0(outputpath, "optrep_total",paste(year[pp], month[kk], sep = "."),"unmasked.Rdata"))
    
    
    
    parnames <- unique(names(rep$jointPrecision[1,]))
    for (i in 1:length(parnames)) {
      eval(parse(text=(paste("parameters$",parnames[i]," <- c(rep$par.fixed,rep$par.random)[names(c(rep$par.fixed,rep$par.random))==\"",parnames[i],"\"]",sep=""))))}
    N_sample<-300
    r.draws <- rmvn.sparse(N_sample,unlist(parameters),Cholesky(rep$jointPrecision),prec=TRUE)
    output.list <- list()
    for (i in 1:N_sample) {output.list[[i]] <- obj$report(r.draws[i,])}
    
    ctable <- list()
    ctable$names <- covariate.names
    ctable$post_mean_slopes <- rep$par.fixed[names(rep$par.fixed)=="static_slopes_Pv"]
    ctable$post_sd_slopes <- sqrt(diag(rep$cov.fixed))[names(rep$par.fixed)=="static_slopes_Pv"]
    ctable$signif <- as.integer(rep$par.fixed[names(rep$par.fixed)=="static_slopes_Pv"]+1.96*sqrt(diag(rep$cov.fixed))[names(rep$par.fixed)=="static_slopes_Pv"] < 0 | rep$par.fixed[names(rep$par.fixed)=="static_slopes_Pv"]-1.96*sqrt(diag(rep$cov.fixed))[names(rep$par.fixed)=="static_slopes_Pv"] > 0)
    ctable <- data.frame(ctable)
    write.csv(ctable,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_covariate_slopes_total_monthly.csv"))
    
    posterior_cases <- matrix(0,nrow=bigN,ncol=N_sample)
    for (i in 1:N_sample) {posterior_cases[,i] <- output.list[[i]]$predicted_surface_malaria_Pv}
    median_cases <- numeric(bigN)
    for (i in 1:bigN) {median_cases[i] <- quantile(posterior_cases[i,]+log(1000),0.5)}
    reference.image[in.country] <- median_cases
    writeRaster(reference.image,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_log_median_cases_total_per_1000_PYO.tif"),overwrite=TRUE)
    
    writeRaster(exp(reference.image),file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_median_cases_total_per_1000_PYO.tif"),overwrite=TRUE)
    
    #test
    test5<-raster(paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_log_median_cases_total_per_1000_PYO.tif"))
    plot(exp(test5));rm(test5)
    
    lower95_cases <- numeric(bigN)
    for (i in 1:bigN) {lower95_cases[i] <- quantile(posterior_cases[i,]+log(1000),0.025)}
    reference.image[in.country] <- lower95_cases
    writeRaster(reference.image,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_log_lower95_cases_total_per_1000_PYO.tif"),overwrite=TRUE)
    
    upper95_cases <- numeric(bigN)
    for (i in 1:bigN) {upper95_cases[i] <- quantile(posterior_cases[i,]+log(1000),0.975)}
    reference.image[in.country] <- upper95_cases
    writeRaster(reference.image,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_log_upper95_cases_total_per_1000_PYO.tif"),overwrite=TRUE)
    
    iqr_cases <- numeric(bigN)
    for (i in 1:bigN) {iqr_cases[i] <- quantile(posterior_cases[i,]+log(1000),0.75)-quantile(posterior_cases[i,]+log(1000),0.25)}
    reference.image[in.country] <- iqr_cases
    writeRaster(reference.image,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_iqr_log_cases_total_per_1000_PYO.tif"),overwrite=TRUE)
    
    reference.image[in.country] <- median_cases+log(population)-log(1000)
    writeRaster(reference.image,file=paste0(outputpath,paste(year[pp], month[kk], sep = "."),"_posterior_log_median_total_cases_total_per_gridcell.tif"),overwrite=TRUE)
    
    #EXCEEDANCE PROB MAPS
    pop_raster<-raster("Common files/Population/SEN_population_v1_0_gridded_1km.tif")
    out_raster <- pop_raster
    #create mask based on pop
    cutoff <- 0 # use cutoff since we use facebook pop
    pop_mask<-pop_raster
    values(pop_mask)[values(pop_mask) < cutoff ] <- 0
    values(pop_mask)[values(pop_mask) > 0] <- 1
    plot(pop_mask)
    pop_mask_na <- pop_mask
    values(pop_mask_na)[values(pop_mask) == 0] <- NA
    #
    #   ###get median, IQR etc.
    rate_full<-posterior_cases
    #
    #   ##do exceedance surfaces
    exc_plot <- function(exc_raster){
      r_points = rasterToPoints(exc_raster)
      r_df = data.frame(r_points)
      r_df$cuts=cut(r_df$prob,breaks=c(-0.1, 0.1, 0.5, 0.75, 1.1))
      p <- ggplot(data=r_df) +
        geom_tile(aes(x=x,y=y,fill=cuts)) +
        scale_fill_brewer("prob",type = "seq") +
        coord_equal() +
        theme_bw() +
        theme(panel.grid.major = element_blank()) +
        xlab("Longitude") + ylab("Latitude")
      print(p)
    }
    #
    names(out_raster) <- "prob"
    #
    #   ##function to make exceedance surfaces
    upper <- 5
    lower <- 1
    exc_above <- function(upper){
      rasterval <- rep(NA, ncell(out_raster))
      upper_prob <- apply(rate_full * 1000, 1, function(row) sum(row > upper) / N_sample)
      rasterval[in.country] <- upper_prob
      upper_raster <- out_raster
      values(upper_raster) <- rasterval
      values(upper_raster)[values(pop_mask) == 0] <- 0
      return(upper_raster)
    }
    
    exc_below <- function(lower){
      rasterval2 <- rep(NA, ncell(out_raster))
      lower_prob <- apply(rate_full * 1000, 1, function(row) sum(row < lower) / N_sample)
      rasterval2[in.country] <- lower_prob
      lower_raster <- out_raster
      values(lower_raster) <-  rasterval2
      values(lower_raster)[values(pop_mask) == 0] <- 1
      return(lower_raster)
    }
    ##check plot
    # exc_plot(exc_above(25))
    # exc_plot(exc_below(10))
    
    upper <- 25
    writeRaster(exc_above(upper), paste0(outputpath,paste(year[pp], month[kk], sep = "."),"upper_", upper, ".tif"),overwrite=TRUE)
    lower <- 5
    writeRaster(exc_below(lower), paste0(outputpath,paste(year[pp], month[kk], sep = "."),"lower_", lower, ".tif"),overwrite=TRUE)
    lower <- 10
    writeRaster(exc_below(lower), paste0(outputpath,paste(year[pp], month[kk], sep = "."),"lower_", lower, ".tif"),overwrite=TRUE)
    lower <- 25
    writeRaster(exc_below(lower), paste0(outputpath,paste(year[pp], month[kk], sep = "."),"lower_", lower, ".tif"),overwrite=TRUE)
    #
    ##checks
    # raster_files <- list.files(outputpath, pattern = '.tif$',full.names =TRUE) #use pattern = '.tif$' or something else if you have multiple files in this folder
    #   rList <- list() # to save raster values
    #   for(i in 1:length(raster_files)){
    #     rList[[i]]<- raster(raster_files[i])
    #   }
    #   par(mfrow=c(2,2))
    # plot(exp(rList[[1]]),main="iqr")
    # plot(exp(rList[[2]]),main="lower")
    # plot(exp(rList[[3]]),main="median")
    # plot(exp(rList[[4]]),main="mediancountgrid")
    # spplot(rList[[10]],main="Aprupper25")
    # spplot(exp(rList[[3]]),main="median")
    #
    #   #count number of cases
    #   predcount <- raster(paste0(outputpath, month[kk],"_posterior_log_median_total_cases_total_per_gridcell.tif"))
    #   predcount <-exp(predcount)
    #   sum(values(predcount),na.rm=TRUE)
    #
  }#end loop
}







#++++++++++++++++++++++++++++++++++++++++++++++++++
#+# Write Out Key Raster Summaries

library(sparseMVN)
library(Matrix)
r.draws <- rmvn.sparse(2,unlist(parameters),Cholesky(rep$jointPrecision),prec=TRUE)
for (i in 1:2) {
  baseline_replicants[[length(baseline_replicants)+1]] <- list()
  outputs <- obj$report(r.draws[i,])
  baseline_replicants[[length(baseline_replicants)]]$field.draws <- outputs$predicted_surface_malaria
  baseline_replicants[[length(baseline_replicants)]]$staticfield.draws <- outputs$static_field
  baseline_replicants[[length(baseline_replicants)]]$gp.draws <- outputs$baseline_field
  baseline_replicants[[length(baseline_replicants)]]$covar.draws <- outputs$static_field_offsets
  baseline_replicants[[length(baseline_replicants)]]$catchments <- outputs$catchments
  baseline_replicants[[length(baseline_replicants)]]$maxpred <- apply(abs(outputs$full_cov_preds),1,which.max)
  baseline_replicants[[length(baseline_replicants)]]$maxpred_pos <- apply((outputs$full_cov_preds),1,which.max)
  baseline_replicants[[length(baseline_replicants)]]$maxpred_neg <- apply((outputs$full_cov_preds),1,which.min)
  baseline_replicants[[length(baseline_replicants)]]$maxpredsign <- sign(outputs$full_cov_preds)[cbind(1:bigN,apply(abs(outputs$full_cov_preds),1,which.max))]
  baseline_replicants[[length(baseline_replicants)]]$baseline_log_masses <- r.draws[i,names(unlist(parameters))=="log_masses.log_masses"]
  baseline_replicants[[length(baseline_replicants)]]$slopes <- r.draws[i,names(unlist(parameters))=="static_slopes.static_slopes"]
  baseline_replicants[[length(baseline_replicants)]]$par.fixed <- rep$par.fixed
  baseline_replicants[[length(baseline_replicants)]]$cov.fix <- rep$cov.fixed
}
save(baseline_replicants,file="baseline_replicants.dat")



field.draws <- gp.draws <- covar.draws <- staticfield.draws <- matrix(NA,ncol=length(baseline_replicants),nrow=bigN)
for (i in 1:length(baseline_replicants)) {
  field.draws[,i] <- baseline_replicants[[i]]$field.draws
  gp.draws[,i] <- baseline_replicants[[i]]$gp.draws
  covar.draws[,i] <- baseline_replicants[[i]]$covar.draws
  staticfield.draws[,i] <- baseline_replicants[[i]]$staticfield.draws
}
pointwise.mean.caserate <- apply(field.draws,1,mean)
pointwise.stddev.caserate <- apply(field.draws,1,sd)
pointwise.mean.gp <- apply(gp.draws,1,mean)
pointwise.mean.covar <- apply(covar.draws,1,mean)
pointwise.mean.covarstatic <- apply(staticfield.draws,1,mean)
pointwise.mean.cases <- log(population)+pointwise.mean.caserate
buffer.image[in.country] <- pointwise.mean.caserate
writeRaster(buffer.image,file="outputs/final_baseline.tif",overwrite=TRUE)
buffer.image[in.country] <- pointwise.stddev.caserate
writeRaster(buffer.image,file="outputs/final_baseline_stddev.tif",overwrite=TRUE)
buffer.image[in.country] <- pointwise.mean.gp
writeRaster(buffer.image,file="outputs/final_baseline_gp.tif",overwrite=TRUE)
buffer.image[in.country] <- pointwise.mean.covar
writeRaster(buffer.image,file="outputs/final_baseline_covs.tif",overwrite=TRUE)
buffer.image[in.country] <- pointwise.mean.covarstatic
writeRaster(buffer.image,file="outputs/final_baseline_staticcovs.tif",overwrite=TRUE)
buffer.image[in.country] <- pointwise.mean.cases
writeRaster(buffer.image,file="outputs/final_baseline_counts.tif",overwrite=TRUE)

exceedance.prob <- apply((field.draws>log(1/1000)),1,mean)
buffer.image[in.country] <- exceedance.prob
writeRaster(buffer.image,file="outputs/final_baseline_prob_exceed_1_per_1000_PYO.tif",overwrite=TRUE)
exceedance.prob <- apply((field.draws>log(50/1000)),1,mean)
buffer.image[in.country] <- exceedance.prob
writeRaster(buffer.image,file="outputs/final_baseline_prob_exceed_50_per_1000_PYO.tif",overwrite=TRUE)
nonexceedance.prob <- apply((field.draws<log(0.1/1000)),1,mean)
buffer.image[in.country] <- nonexceedance.prob
writeRaster(buffer.image,file="outputs/final_baseline_prob_nonexceed_1_per_10000_PYO.tif",overwrite=TRUE)

pop.at.risk <- colSums((field.draws>log(1/1000))*matrix(rep(population,length(baseline_replicants)),ncol=length(baseline_replicants)))
pop.at.risk.median <- median(pop.at.risk)
pop.at.risk.lower <- quantile(pop.at.risk,0.025)
pop.at.risk.upper <- quantile(pop.at.risk,0.975)
pop.output <- as.matrix(c(pop.at.risk.median,pop.at.risk.lower,pop.at.risk.upper))
names(pop.output) <- c("Median","Lower 95%","Upper 95%")
write.csv(pop.output,file="outputs/population_at_risk_national_1_per_1000_PYO.csv")

reference.coords <- in.country.coords
admin1 <- readOGR("adm/hti_admbnda_adm1_cnigs_20181129.shp")
pop.at.risk.summary <- list()
for (i in 1:length(admin1)) {
  in.admin.sector <- numeric(bigN)
  for (j in 1:length(admin1@polygons[[i]]@Polygons)) {
    in.admin.sector <- in.admin.sector + point.in.polygon(reference.coords[,1],reference.coords[,2],admin1@polygons[[i]]@Polygons[[j]]@coords[,1],admin1@polygons[[i]]@Polygons[[j]]@coords[,2])
  }
  pop.at.risk <- colSums((field.draws>log(1/1000))*matrix(rep(population,length(baseline_replicants)),ncol=length(baseline_replicants))*matrix(rep(in.admin.sector,length(baseline_replicants)),ncol=length(baseline_replicants)))
  pop.at.risk.median <- median(pop.at.risk)
  pop.at.risk.lower <- quantile(pop.at.risk,0.025)
  pop.at.risk.upper <- quantile(pop.at.risk,0.975)
  pop.at.risk.summary[[i]] <- c(pop.at.risk.median,pop.at.risk.lower,pop.at.risk.upper)  
}
pop.at.risk.summary <- do.call(rbind,pop.at.risk.summary)
pop.at.risk.summary <- cbind(as.character(admin1@data$ADM1_FR),as.character(pop.at.risk.summary[,1]),as.character(pop.at.risk.summary[,2]),as.character(pop.at.risk.summary[,3]))
colnames(pop.at.risk.summary) <- c("Department","Median","Lower 95%","Upper 95%")
write.csv(pop.output,file="outputs/population_at_risk_department_1_per_1000_PYO.csv")
admin1@data$MedianPopAtRisk <- as.numeric(pop.at.risk.summary[,2])
writeOGR(admin1, ".", "outputs/popatrisk_department_1_per_1000_PYO", driver="ESRI Shapefile",overwrite_layer = TRUE)

admin2 <- readOGR("adm/hti_admbnda_adm2_cnigs_20181129.shp")
pop.at.risk.summary <- list()
for (i in 1:length(admin2)) {
  in.admin.sector <- numeric(bigN)
  for (j in 1:length(admin2@polygons[[i]]@Polygons)) {
    in.admin.sector <- in.admin.sector + point.in.polygon(reference.coords[,1],reference.coords[,2],admin2@polygons[[i]]@Polygons[[j]]@coords[,1],admin2@polygons[[i]]@Polygons[[j]]@coords[,2])
  }
  pop.at.risk <- colSums((field.draws>log(1/1000))*matrix(rep(population,length(baseline_replicants)),ncol=length(baseline_replicants))*matrix(rep(in.admin.sector,length(baseline_replicants)),ncol=length(baseline_replicants)))
  pop.at.risk.median <- median(pop.at.risk)
  pop.at.risk.lower <- quantile(pop.at.risk,0.025)
  pop.at.risk.upper <- quantile(pop.at.risk,0.975)
  pop.at.risk.summary[[i]] <- c(pop.at.risk.median,pop.at.risk.lower,pop.at.risk.upper)  
}
pop.at.risk.summary <- do.call(rbind,pop.at.risk.summary)
pop.at.risk.summary <- cbind(as.character(admin2@data$ADM1_FR),as.character(pop.at.risk.summary[,1]),as.character(pop.at.risk.summary[,2]),as.character(pop.at.risk.summary[,3]))
colnames(pop.at.risk.summary) <- c("Commune","Median","Lower 95%","Upper 95%")
write.csv(pop.output,file="outputs/population_at_risk_commune_1_per_1000_PYO.csv")
admin2@data$MedianPopAtRisk <- as.numeric(pop.at.risk.summary[,2])
writeOGR(admin2, ".", "outputs/popatrisk_commune_1_per_1000_PYO", driver="ESRI Shapefile",overwrite_layer = TRUE)

nearest.hf <- matrix(NA,nrow=bigN,ncol=length(baseline_replicants))
for (i in 1:length(baseline_replicants)) {
  nearest.hf[,i] <- apply(as.matrix(baseline_replicants[[i]]$catchments),2,which.max)
}
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
nearest.hf <- apply(nearest.hf,1,getmode)
buffer.image[in.country] <- runif(nHFs)[nearest.hf]
writeRaster(buffer.image,file="outputs/nearest_hf_visualization.tif",overwrite=TRUE)

length.nontrivial <- function(x) {length(which(x>0.1))}
n.local.hfs <- matrix(NA,nrow=bigN,ncol=length(baseline_replicants))
for (i in 1:length(baseline_replicants)) {
  n.local.hfs[,i] <- apply(as.matrix(baseline_replicants[[i]]$catchments),2,length.nontrivial)
}
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
n.local.hfs <- apply(n.local.hfs,1,getmode)
buffer.image[in.country] <- n.local.hfs
writeRaster(buffer.image,file="outputs/nlocal_hf_visualization.tif",overwrite=TRUE)

maxpredpos <- maxpredneg <- matrix(NA,ncol=length(baseline_replicants),nrow=bigN)
for (i in 1:length(baseline_replicants)) {
  maxpredpos[,i] <- baseline_replicants[[i]]$maxpred_pos
  maxpredneg[,i] <- baseline_replicants[[i]]$maxpred_neg
}
buffer.image[in.country] <- apply(maxpredpos,1,getmode)
writeRaster(buffer.image,file="outputs/final_most_important_pos_covariate.tif",overwrite=TRUE)
buffer.image[in.country] <- apply(maxpredneg,1,getmode)
writeRaster(buffer.image,file="outputs/final_most_important_neg_covariate.tif",overwrite=TRUE)

maxpred <- maxpredsign <- matrix(NA,ncol=length(baseline_replicants),nrow=bigN)
for (i in 1:length(baseline_replicants)) {
  maxpred[,i] <- baseline_replicants[[i]]$maxpred
  maxpredsign[,i] <- baseline_replicants[[i]]$maxpredsign
}
buffer.image[in.country] <- apply(maxpred,1,getmode)
writeRaster(buffer.image,file="outputs/final_most_important_covariate.tif",overwrite=TRUE)
buffer.pix <- apply(maxpred,1,getmode)
for (i in 1:bigN) {buffer.pix[i] <- getmode(maxpredsign[i,maxpred[i,]==buffer.pix[i]])}
buffer.image[in.country] <- buffer.pix
writeRaster(buffer.image,file="outputs/final_sign_most_important_covariate.tif",overwrite=TRUE)

mean_slopes <- matrix(0,nrow=Ncovariates,ncol=length(baseline_replicants))
for (i in 1:(length(baseline_replicants)/2)) {mean_slopes[,i] <- baseline_replicants[[i*2-1]]$par.fixed[2:13]}
mean_slopes_mean <- rowMeans(mean_slopes)
mean_slopes_stddev <- sqrt(rowMeans(mean_slopes*mean_slopes)-rowMeans(mean_slopes)^2)
issignif <- as.integer(mean_slopes_mean+mean_slopes_stddev*3 < 0 | mean_slopes_mean-mean_slopes_stddev*3 > 0)
cov.stat <- cbind(covariate.names,as.character(mean_slopes_mean),as.character(mean_slopes_stddev),as.character(issignif))
colnames(cov.stat) <- c("Covariate Name","Post Mean Slope","Post Std Dev Slope","3 sig signif.?")
write.csv(cov.stat,file="outputs/covstats.csv")

save.image("postfit_static.dat")

agg.pops <- matrix(NA,nrow=nHFs,ncol=length(baseline_replicants))
for (i in 1:length(baseline_replicants)) {agg.pops[,i] <- as.numeric(baseline_replicants[[i]]$catchments%*%(population*treatment))}
med.pop <- upp.pop <- low.pop <- numeric(nHFs)
for (i in 1:nHFs) {
  med.pop[i] <- quantile(agg.pops[i,],0.5)
  low.pop[i] <- quantile(agg.pops[i,],0.025)
  upp.pop[i] <- quantile(agg.pops[i,],0.975)
}
pop.summ <- cbind(1:nHFs,med.pop,low.pop,upp.pop)
agg.pops <- matrix(NA,nrow=nHFs,ncol=length(baseline_replicants))
for (i in 1:length(baseline_replicants)) {agg.pops[,i] <- as.numeric(baseline_replicants[[i]]$catchments%*%(population))}
med.pop <- upp.pop <- low.pop <- numeric(nHFs)
for (i in 1:nHFs) {
  med.pop[i] <- quantile(agg.pops[i,],0.5)
  low.pop[i] <- quantile(agg.pops[i,],0.025)
  upp.pop[i] <- quantile(agg.pops[i,],0.975)
}
pop.summ <- cbind(pop.summ,med.pop,low.pop,upp.pop)
colnames(pop.summ) <- c("Aggregated HF ID Number","Posterior Median Catchment Pop Est (Will Seek Treatment)","Posterior 2.5% Catchment (Will Seek Treatment)","Upper 97.5% Catchment Pop Est (Will Seek Treatment)","Posterior Median Catchment Pop Est (Ignoring Treatment Seeking)","Posterior 2.5% Catchment (Ignoring Treatment Seeking)","Upper 97.5% Catchment Pop Est (Ignoring Treatment Seeking)")
write.csv(pop.summ,file="outputs/aggregated_catchment_pops.csv")

hf.agg <- cbind(as.character(hf.ids.reporting),as.character(hf.longlats.reporting[,1]),as.character(hf.longlats.reporting[,2]),as.character(clusterCut),as.character(hf.longlats.clustered[clusterCut,1]),as.character(hf.longlats.clustered[clusterCut,2]))
colnames(hf.agg) <- c("HF ID Code","HF Long","HF Lat","Aggregated HF ID Number","Agg HF Long","Agg HF Lat")
write.csv(hf.agg,file="outputs/hf_aggregation_codes.csv")

## Catchment visualisation

cases <- exp(raster("outputs/final_baseline_counts.tif")[in.country])*treatment
mean.catchments <- matrix(0,nrow=nHFs,ncol=bigN)
for (i in 1:length(baseline_replicants)) {mean.catchments <- mean.catchments+as.matrix(baseline_replicants[[i]]$catchments)/length(baseline_replicants)}

for (i in 1:nHFs) {
  catchx <- mean.catchments[i,]*cases
  catchx <- catchx/sum(catchx)
  values(buffer.image) <- NA
  buffer.image[in.country] <- catchx
  writeRaster(buffer.image,file=paste0("outputs/prob_case_origin_given_reported_at_aggregated_HF_number_",sprintf("%03i",i),".tif"))
}

all.complete.journeys.list <- list()
all.complete.journeys.weights.list <- list()
all.complete.journeys.ids.list <- list()

for (facility.num in 1:nHFs) {
  
  expected.casepix <- mean.catchments[facility.num,]*cases
  casepix <- which(expected.casepix > 0)
  weights <- expected.casepix[expected.casepix > 0]
  # normweights <- weights/sum(weights)
  casepix <- casepix[which(weights > 0.1)]
  weights <- weights[which(weights > 0.1)]
  # weights <- sort(weights,decreasing=TRUE)
  # normweights <- sort(normweights,decreasing=TRUE)
  # casepix <- casepix[cumsum(weights)<0.99]
  # weights <- weights[cumsum(normweights)<0.99]
  # normweights <- normweights[cumsum(normweights)<0.99]
  
  if (length(casepix) > 1) {
    
    journeys <- list()
    for (i in 1:length(weights)) {
      buffer <- shortestPath(T.GC, as.numeric(xyFromCell(reference.image,in.country[casepix[i]])), as.numeric(hf.longlats.clustered[facility.num,]), output = "SpatialLines")@lines[[1]]
      buffer@ID <- as.character(i)
      journeys[[i]] <- buffer
    }
    new.journeys <- list()
    new.weights <- list()
    for (i in 1:length(journeys)) {
      if (length(journeys[[i]]@Lines[[1]]@coords)>2) {
        new.journeys[[length(new.journeys)+1]] <- journeys[[i]]
        new.weights[[length(new.weights)+1]] <- weights[i]
      }
    }
    journeys <- new.journeys
    weights <- as.numeric(new.weights)
    
    complete.journeys <- list()
    complete.journeys.weights <- list()
    complete.journeys[[1]] <- journeys[[1]]@Lines[[1  Error when reading the variable: 'log_shrinkage_temporal_slopes'. Please check data and parameters.
    ]]@coords[1:2,]
    complete.journeys.weights[[1]] <- weights[1]
    for (i in 1:length(journeys)) {
      if (length(journeys[[i]]@Lines[[1]]@coords)>2) {
        for (j in 1:(length(journeys[[i]]@Lines[[1]]@coords[,1])-1)) {
          in.complete.journeys <- 0
          for (k in 1:length(complete.journeys)) {
            if (prod(journeys[[i]]@Lines[[1]]@coords[(j):(j+1),]==complete.journeys[[k]])) {
              complete.journeys.weights[[k]] <- complete.journeys.weights[[k]] + weights[i]
              in.complete.journeys <- 1
            }
          }
          if (in.complete.journeys==0) {
            complete.journeys[[length(complete.journeys)+1]] <- journeys[[i]]@Lines[[1]]@coords[(j):(j+1),]
            complete.journeys.weights[[length(complete.journeys.weights)+1]] <- weights[i]
          }
        }}
    }
    complete.journeys.sl <- list()
    for (i in 1:length(complete.journeys)) {
      complete.journeys.sl[[i]] <- Lines(Line(complete.journeys[[i]]), ID=as.numeric(facility.num*1000+i))
    }
    all.complete.journeys.list[[length(all.complete.journeys.list)+1]] <- complete.journeys.sl
    all.complete.journeys.weights.list[[length(all.complete.journeys.weights.list)+1]] <- as.numeric(complete.journeys.weights)
    all.complete.journeys.ids.list[[length(all.complete.journeys.ids.list)+1]] <- rep(facility.num,length(complete.journeys.weights))
    cat(facility.num,"\n")
  }}
sl_obj <- SpatialLines(unlist(all.complete.journeys.list))
ids <- data.frame()
for (i in (1:length(sl_obj))) {
  id <- data.frame(sl_obj@lines[[i]]@ID)
  ids <- rbind(ids, id)
}
colnames(ids)[1] <- "linkId"
row.names(ids) <- ids$linkId
splndf <- SpatialLinesDataFrame(sl_obj, data = ids, match.ID = TRUE)
splndf$weights <- log(unlist(all.complete.journeys.weights.list))
HFids <- runif(450)
splndf$cIds <- HFids[unlist(all.complete.journeys.ids.list)]
library(rgdal)
writeOGR(splndf, dsn="." ,layer="journeys",driver="ESRI Shapefile",overwrite_layer = TRUE)
hfcols <- cbind(xyFromCell(reference.image,cellFromXY(reference.image,hf.longlats.clustered)),HFids)
colnames(hfcols) <- c("long","lat","colid")
write.csv(hfcols,"journeyHFs.csv")

all.complete.journeys.list <- list()
all.complete.journeys.weights.list <- list()
all.complete.journeys.ids.list <- list()

for (facility.num in 1:nHFs) {
  
  expected.casepix <- mean.catchments[facility.num,]
  casepix <- which(expected.casepix > 0)
  weights <- (population*treatment)[casepix]
  normweights <- weights/sum(weights)
  normweights <- sort(normweights,decreasing=TRUE)
  casepix <- casepix[cumsum(normweights)<0.90]
  normweights <- normweights[cumsum(normweights)<0.90]
  weights <- normweights
  
  if (length(casepix) > 1) {
    
    journeys <- list()
    for (i in 1:length(weights)) {
      buffer <- shortestPath(T.GC, as.numeric(xyFromCell(reference.image,in.country[casepix[i]])), as.numeric(hf.longlats.clustered[facility.num,]), output = "SpatialLines")@lines[[1]]
      buffer@ID <- as.character(i)
      journeys[[i]] <- buffer
    }
    new.journeys <- list()
    new.weights <- list()
    for (i in 1:length(journeys)) {
      if (length(journeys[[i]]@Lines[[1]]@coords)>2) {
        new.journeys[[length(new.journeys)+1]] <- journeys[[i]]
        new.weights[[length(new.weights)+1]] <- weights[i]
      }
    }
    journeys <- new.journeys
    weights <- as.numeric(new.weights)
    
    complete.journeys <- list()
    complete.journeys.weights <- list()
    complete.journeys[[1]] <- journeys[[1]]@Lines[[1]]@coords[1:2,]
    complete.journeys.weights[[1]] <- 0#weights[1]
    for (i in 1:length(journeys)) {
      if (length(journeys[[i]]@Lines[[1]]@coords)>2) {
        for (j in 1:(length(journeys[[i]]@Lines[[1]]@coords[,1])-1)) {
          in.complete.journeys <- 0
          for (k in 1:length(complete.journeys)) {
            if (prod(journeys[[i]]@Lines[[1]]@coords[(j):(j+1),]==complete.journeys[[k]])) {
              complete.journeys.weights[[k]] <- complete.journeys.weights[[k]] + weights[i]
              in.complete.journeys <- 1
            }
          }
          if (in.complete.journeys==0) {
            complete.journeys[[length(complete.journeys)+1]] <- journeys[[i]]@Lines[[1]]@coords[(j):(j+1),]
            complete.journeys.weights[[length(complete.journeys.weights)+1]] <- weights[i]
          }
        }}
    }
    complete.journeys.sl <- list()
    for (i in 1:length(complete.journeys)) {
      complete.journeys.sl[[i]] <- Lines(Line(complete.journeys[[i]]), ID=as.numeric(facility.num*1000+i))
    }
    all.complete.journeys.list[[length(all.complete.journeys.list)+1]] <- complete.journeys.sl
    all.complete.journeys.weights.list[[length(all.complete.journeys.weights.list)+1]] <- as.numeric(complete.journeys.weights)
    all.complete.journeys.ids.list[[length(all.complete.journeys.ids.list)+1]] <- rep(facility.num,length(complete.journeys.weights))
    cat(facility.num,"\n")
  }}
sl_obj <- SpatialLines(unlist(all.complete.journeys.list))
ids <- data.frame()
for (i in (1:length(sl_obj))) {
  id <- data.frame(sl_obj@lines[[i]]@ID)
  ids <- rbind(ids, id)
}
colnames(ids)[1] <- "linkId"
row.names(ids) <- ids$linkId
splndf <- SpatialLinesDataFrame(sl_obj, data = ids, match.ID = TRUE)
splndf$weights <- log(unlist(all.complete.journeys.weights.list))
splndf$cIds <- HFids[unlist(all.complete.journeys.ids.list)]
library(rgdal)
writeOGR(splndf, dsn="." ,layer="alljourneys",driver="ESRI Shapefile",overwrite_layer = TRUE)












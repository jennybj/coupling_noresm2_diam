using DelimitedFiles
using Printf

include("c:/econ417julia/io4.jl")
include("c:/julia/sort1.jl")
include("c:/climate/createarray1.jl")
include("c:/climate/stats1.jl")

const missingvalue = -99.99
const missingintvalue = -99
const ncells = 19240
const toler = 1e-8

function missing(x;missingvalue=-99.99,missingintvalue=-99)

   if (typeof(x) == Float64)
      missing = (x == missingvalue)
   elseif (typeof(x) == Int64)
      missing = (x == missingintvalue)   
   else
      wait("Unrecognized type in missing.")
   end      
   
end   

function samestring(s1,s2)

   s1trim = trim(s1)
   s2trim = trim(s2)
   
   samestring = (s1trim == s2trim)
   
end   

function trim(s)

   n = length(s)
   lastchar = 0
   
   for i in n:-1:1
      if (s[i] !== ' ') 
         lastchar = i
         break
      end
   end
   
   if lastchar > 0
      strim = s[1:lastchar]      
   else
      strim = ""
   end
   
   return strim
   
end

function findinlist(list,name)

   n = length(list)
   
   index = 0
   
   for i = 1:n
      if samestring(list[i],name)
         index = i
         break
      end
   end
   
   found = (index > 0)
   
   return found,index
   
end            

function readdata(filename)

   io = open(filename,"r")
   datamatrix = readdlm(io,',')
   close(io)
   
   return datamatrix
   
end

function sharebounds(s;toler=1e-12)

   if (s <= toler)
      sbound = toler
   elseif (s >= 1-toler)
      sbound = 1 - toler
   else
      sbound = s
   end   
   
   return sbound
   
end   

#filename = "c:/Dropbox/noresm/population_growth/shares_method/nord40_gpw_populations.csv"
#datamatrix = readdata(filename)
io = open("regpop3.popkeep","r")
popregi = fill(0.0,ncells,4)
for i in 1:ncells
#   popregi[i,1:4] = datamatrix[i+1,5:8]
   d,popregi[i,:] = readio(io,(1,"r4")) 
end   

writeio(stdout,((4,20.12),),minimum(popregi[:,1]),minimum(popregi[:,2]),
   minimum(popregi[:,3]),minimum(popregi[:,4]),callwait=false,write=true)

writeio(stdout,((4,20.6),),sum(popregi[:,1]),sum(popregi[:,2]),
   sum(popregi[:,3]),sum(popregi[:,4]),callwait=false,write=true)
   
filename = "c:/Dropbox/noresm/population_growth/shares_method/undp_wide.csv"
datamatrix = readdata(filename)
numundp = size(datamatrix)[2] - 1
writeio(stdout,("numundp: ",6),numundp)
undpcountries = convert.(String,datamatrix[1,2:numundp+1])
for i in 1:numundp
   if samestring("Guinea-Bissau",undpcountries[i])
      undpcountries[i] = "Guinea Bissau"
      break
   end
end        

nyears = size(datamatrix)[1] - 1
open("regpop4.undp","w") do io
   for i in 1:numundp
      popdata = convert.(Float64,datamatrix[2:nyears+1,i+1])
      writeio(io,(5,"  ",40,15.5),i,undpcountries[i],popdata)
   end
end   

yearrangeundp = datamatrix[2,1]:datamatrix[nyears+1,1]
writeio(stdout,("UNDP year range: ",(2,6)),yearrangeundp[1],yearrangeundp[nyears])

popundp = createoffsetarray(0.0,(1:numundp,yearrangeundp))
popundp[1:numundp,yearrangeundp] .= convert.(Float64,datamatrix[2:nyears+1,2:numundp+1]')

grateundp = createoffsetarray(0.0,(1:numundp,yearrangeundp))
for i in 1:numundp
   for j in yearrangeundp[1]+1:yearrangeundp[nyears]
      grateundp[i,j] = popundp[i,j]/popundp[i,j-1] - 1
      writeio(stdout,((2,6),(3,18.8)),i,j,popundp[i,j],popundp[i,j-1],grateundp[i,j],
         callwait=false,write=false)
   end
   wait(undpcountries[i],condition=false)
end  

wait("Done writing to regpop4.undp.")

countryi = fill("",ncells)
lati = fill(0,ncells)
loni = fill(0,ncells)
areai = fill(0.0,ncells)
rigi = fill(0.0,ncells)
avgtempi = fill(0.0,ncells)
popregi1990 = fill(0.0,ncells)
gdpregi1990 = fill(0.0,ncells)
gdpperregi1990 = fill(0.0,ncells)

#filename = "c:/Dropbox/noresm/endogenous_grid_method/Input Files/parse2.gin5"
#filename = "c:/Dropbox/noresm/population_growth/nordhaus_update/parse2.gin5"
io = open("parse2.gin6","r")
for i in 1:ncells

   d,lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
      gdpregi1990[i],gdpperregi1990[i] = readio(io,(3,"b3","a40",6))
      
#   if (popregi1990[i] < 0.0005) 
#      writeio(stdout,((3,6),"  ",10,"  ",(3,15.8),13.6),i,lati[i],loni[i],countryi[i],
#         popregi1990[i],gdpregi1990[i],gdpperregi1990[i],popregi[i,:],
#         callwait=true,write=true)
#   end

end
close(io)

wait("Done reading parse2.gin6.")

blank = fill(" "^2,ncells)
writearrays(stdout,(7,(2,6),2,15,2,(3,25.10)),lati,loni,blank,countryi,
   blank,popregi[:,1],popregi1990,gdpregi1990,nmult=25,addzero=true,write=false)
   
countries = fill("",1)
numregions = fill(0,1)
countriespop = fill(0.0,1,4)
countryindexi = fill(0,ncells)
countries[1] = countryi[1]
numregions[1] = 1
countriespop[1,1:4] = popregi[1,1:4]
countryindexi[1] = 1
for i in 2:ncells
   global countriespop
   found,ix = findinlist(countries,countryi[i])
   if !found
      push!(countries,countryi[i])
      push!(numregions,1)
      countriespop = [countriespop;popregi[i,:]']
      countryindexi[i] = length(countries)
   else
      numregions[ix] += 1
      countriespop[ix,:] .+= popregi[i,:]
      countryindexi[i] = ix
   end      
   writeio(stdout,(6,6,"  ",20,"  ",15.5),i,countryindexi[i],countryi[i],
      popregi[i,:],callwait=multpl(i,25),write=false)   
end 

countryindex = sort(countries)
numcountries = length(countries)
            
writeio(stdout,(6,10),length(countries),sum(numregions))
writeio(stdout,((4,20.6),),sum(popregi[:,1]),sum(popregi[:,2]),
   sum(popregi[:,3]),sum(popregi[:,4]),callwait=false,write=true)
writeio(stdout,((4,20.6),),sum(countriespop[:,1]),sum(countriespop[:,2]),
   sum(countriespop[:,3]),sum(countriespop[:,4]),callwait=false,write=true)

undpindex = fill(missingintvalue,numcountries)
writeio(stdout,("Countries in the UNDP list not found in the Nordhaus database:",))
for i in 1:numundp
   found,ix = findinlist(countries,undpcountries[i])
   if found
      undpindex[ix] = i 
   else
      writeio(stdout,40,undpcountries[i])
   end
end

wait()

blank = fill(" "^3,numcountries)
writearrays(stdout,(5,3,20,3,6,6,20.8),blank,countries[countryindex],blank,
            numregions[countryindex],undpindex[countryindex],
            countriespop[countryindex,:],nmult=25,write=false,lastwait=true)

nmissingregions = 0
writeio(stdout,("Countries in the Nordhaus database not found in the UNDP list:",))
for i in 1:numcountries
   global nmissingregions
   found,ix = findinlist(undpcountries,countries[i])
   if !found
      writeio(stdout,40,countries[i])
      nmissingregions += numregions[i]
   end
end
writeio(stdout,("Total number of missing regions: ",10),nmissingregions)
writeio(stdout,("Number of missing countries: ",10),sum(undpindex.==missingintvalue))

sharenordi = fill(0.0,ncells,4)
for i in 1:ncells
   sharenordi[i,:] = popregi[i,:]./countriespop[countryindexi[i],:]
   writeio(stdout,(6," ",20," ",6,(2,15.8)),i,countryi[i],countryindexi[i],
      sharenordi[i,:],sharenordi[i,4]-sharenordi[i,1],callwait=multpl(i,25),write=false)
end   

sumshare = fill(0.0,numcountries,4)
for i in 1:numcountries
   ix = countryindex[i]
   for j in 1:4
      sumshare[ix,j] = sum(sharenordi[:,j].*(countryindexi.==ix))
   end   
   writeio(stdout,((2,6)," ",20," ",15.8),i,ix,countries[ix],sumshare[ix,:],
      callwait = multpl(i,25),write=false)
end   

firstyear = 1990
lastyear = 2140
popi = createoffsetarray(missingvalue,(1:ncells,firstyear:lastyear))
for interval in 1:3

   if (interval == 1)
      yearstart = 1990
      yearend = 1994
   elseif (interval == 2)
      yearstart = 1995
      yearend = 1999
   elseif (interval == 3)
      yearstart = 2000
      yearend = 2004
   end
   
   ncut = 0
   for i in 1:ncells
      intercepti = popregi[i,interval]
      slopei = (popregi[i,interval+1] - popregi[i,interval])/5
      for j in 0:4
         popi[i,yearstart+j] = intercepti + slopei*j
      end
      if (interval == 3)
         popi[i,yearend+1] = popregi[i,interval+1]
      end
      if (interval <= 2)   
         writeio(stdout,((2,6),15.8,(3,15.8)),interval,i,popi[i,yearstart:yearend],
            popi[i,yearend]+slopei,popregi[i,interval],popregi[i,interval+1],
            callwait=false,write=(i<=ncut))
      else      
         writeio(stdout,((2,6),15.8,(2,15.8)),interval,i,popi[i,yearstart:yearend+1],
            popregi[i,interval],popregi[i,interval+1],callwait=false,write=(i<=ncut))
      end      
   end
   
   wait("Interval "*string(interval)*" done.",condition=false)
   
end  

writeio(stdout,("Calculating shares...",))

shareyearstart = 2006
shareyearend = 2100
sharei = createoffsetarray(0.0,(1:ncells,firstyear:shareyearend))
for i in 1:ncells
   
   for j in shareyearstart:shareyearend
   
      share1 = sharebounds(sharenordi[i,1])
      share4 = sharebounds(sharenordi[i,4])
      
      if (share4 < share1)
         a = log(share1)
         b = (log(share4) - a)/15
         sharei[i,j] = sharebounds(exp(a + b*(j-1990)))
      else
         a = log(1 - share1)
         b = (log(1 - share4) - a)/15
         sharei[i,j] = sharebounds(1 - exp(a + b*(j-1990)))
      end   
      
      writeio(stdout,((2,6),(5,20.12)),i,j,a,b,sharenordi[i,1],sharenordi[i,4],sharei[i,j],
         write=false)
         
   end
   
   wait("Done with region "*string(i)*" in "*trim(countryi[i])*".",
      condition=false)

end

writeio(stdout,("Done calculating shares.",))

for i in shareyearstart:shareyearend
   sharemin = maximum(sharei[:,i])
   sharemax = minimum(sharei[:,i])
   sumzero = sum(sharei[:,i].<0)
   sumone = sum(sharei[:,i].>1)
   writeio(stdout,(6,(2,25.12),(2,10)),i,sharemin,sharemax,sumzero,sumone,
      callwait=multpl(i,25),write=false)
end

writeio(stdout,("Calculating share sums...",))

sharesums = fill(0.0,numcountries,shareyearstart:shareyearend)
for i in 1:ncells
   ix = countryindexi[i]
   sharesums[ix,shareyearstart:shareyearend] .+= sharei[i,shareyearstart:shareyearend]
   writeio(stdout,((2,8),"  ",20),i,ix,countries[ix],callwait=multpl(i,1000),write=false)
end   

writeio(stdout,("Done calculating share sums.",))

for i in 1:numcountries

   ix = countryindex[i]
   
   for j in shareyearstart:shareyearend
      writeio(stdout,((2,6)," ",20," ",15.8),ix,j,countries[ix],sharesums[ix,j],
         callwait=(j==shareyearend),write=false)
   end
   sharemin = minimum(sharesums[ix,:]) 
   sharemax = maximum(sharesums[ix,:])
   writeio(stdout,(6," ",20," ",7,(2,15.8)),i,countries[ix],numregions[ix],sharemin,sharemax,
      callwait=(i==numcountries),write=true)
      
end      

for i in 1:ncells
   ix = countryindexi[i]
   yearrange = shareyearstart:shareyearend
   sharei[i,yearrange] = sharei[i,yearrange]./sharesums[ix,yearrange]
end   

sharesums = fill(0.0,numcountries,shareyearstart:shareyearend)
for i in 1:ncells
   ix = countryindexi[i]
   sharesums[ix,shareyearstart:shareyearend] .+= sharei[i,shareyearstart:shareyearend]
   writeio(stdout,((2,8),"  ",20),i,ix,countries[ix],callwait=multpl(i,1000),write=false)
end   

writeio(stdout,("Recalculating share sums...",))

sharesums = fill(0.0,numcountries,shareyearstart:shareyearend)
for i in 1:ncells
   ix = countryindexi[i]
   sharesums[ix,shareyearstart:shareyearend] .+= sharei[i,shareyearstart:shareyearend]
   writeio(stdout,((2,8),"  ",20),i,ix,countries[ix],callwait=multpl(i,1000),write=false)
end   

writeio(stdout,("Done recalculating share sums.",))

for i in 1:numcountries

   ix = countryindex[i]
   
   for j in shareyearstart:shareyearend
      writeio(stdout,((2,6)," ",20," ",15.8),ix,j,countries[ix],sharesums[ix,j],
         callwait=(j==shareyearend),write=false)
   end
   sharemin = minimum(sharesums[ix,:]) 
   sharemax = maximum(sharesums[ix,:])
   writeio(stdout,(6," ",20," ",(2,7),(2,15.8)),i,countries[ix],undpindex[ix],numregions[ix],
      sharemin,sharemax,callwait=multpl(i,numcountries),write=true)
      
end      

lastundpyear = yearrangeundp[nyears]
grateproj = createoffsetarray(0.0,(1:numundp,lastundpyear+1:lastyear))
nyearsproj = lastyear - lastundpyear
for i in 1:numundp
   a = grateundp[i,lastundpyear]
   b = -a/nyearsproj
   for j in lastundpyear+1:lastyear
      grateproj[i,j] = a + b*(j-lastundpyear)
      writeio(stdout,((2,15.8),(2,6),15.8),a,b,i,j,grateproj[i,j],
         callwait=(j==lastyear),write=false)
   end
end      

#for i in 1:numcountries
#   ix = countryindex[i]
#   undpix = undpindex[ix]
#   if !missing(undpix)
#      writeio(stdout,((3,6),"  ",20,"  ",20),i,ix,undpindex[ix],countries[ix],
#         undpcountries[undpix],callwait=multpl(i,25),write=true)
#   else     
#      writeio(stdout,((3,6),"  ",20),i,ix,undpindex[ix],countries[ix],
#         callwait=multpl(i,25),write=true)
#   end
#end         

startyear = 2006
endyear = 2140
endundpyear = yearrangeundp[nyears]
writeio(stdout,((3,6),),startyear,endundpyear,endyear,callwait=false,write=true)

countriespopproj = createoffsetarray(0.0,(1:numcountries,startyear:endyear))
gratemissing = 0.0
for i in 1:numcountries

   ix = countryindex[i]
   undpix = undpindex[ix]
   
   if !missing(undpix)

      writeio(stdout,((3,6),"  ",20,"  ",20),i,ix,undpix,countries[ix],
         undpcountries[undpix],callwait=multpl(i,25),write=false)
   
      countriespopproj[ix,startyear] = (1+grateundp[undpix,startyear])*countriespop[ix,4]
      
      for j in startyear+1:endundpyear
         countriespopproj[ix,j] = (1+grateundp[undpix,j])*countriespopproj[ix,j-1]
      end
      
      yearrange = startyear+1:endundpyear
      writearrays(stdout,(6,(2,20.8)),yearrange,grateundp[undpix,yearrange],
         countriespopproj[ix,yearrange],writeindex=false,nmult=0,write=false,lastwait=false)
      
      for j in endundpyear+1:endyear
         countriespopproj[ix,j] = (1+grateproj[undpix,j])*countriespopproj[ix,j-1]
      end

      yearrange = endundpyear+1:endyear
      writearrays(stdout,(6,(2,20.8)),yearrange,grateproj[undpix,yearrange],
         countriespopproj[ix,yearrange],writeindex=false,nmult=0,write=false,lastwait=true)      
      
   else
    
      writeio(stdout,((3,6),"  ",20),i,ix,undpindex[ix],countries[ix],
         callwait=multpl(i,25),write=false)
    
      countriespopproj[ix,startyear] = (1+gratemissing)*countriespop[ix,4]
   
      for j in startyear+1:endyear
         countriespopproj[ix,j] = (1+gratemissing)*countriespopproj[ix,j-1]
      end

      yearrange = startyear+1:endyear
      writearrays(stdout,(6,(2,20.8)),yearrange,fill(gratemissing,length(yearrange)),
         countriespopproj[ix,yearrange],writeindex=false,nmult=25,write=false,lastwait=true)      
      
   end
   
end         

baseyear = 1990
countrysums = createoffsetarray(0.0,(1:numcountries,baseyear:endyear))
worldpop = createoffsetarray(0.0,(baseyear:endyear,))

writeio(stdout,("Calculating regional populations...",))

minpop = 0.001
for i in 1:ncells

   writeio(stdout,10,i,write=multpl(i,2000))
      
   ix = countryindexi[i]
   
   yearrange = startyear:endundpyear
   popi[i,yearrange] = max.(sharei[i,yearrange].*countriespopproj[ix,yearrange],Ref(minpop))
   
   yearrange = endundpyear+1:endyear
   popi[i,yearrange] = max.(sharei[i,endundpyear].*countriespopproj[ix,yearrange],Ref(minpop))
   
   for j in baseyear:startyear-1
     writeio(stdout,(7,"  ",20,"  ",(2,6)," "^40,20.8),i,countryi[i],ix,j,
        popi[i,j],callwait=multpl(j,25),write=false)
   end     
   
   for j in startyear:endundpyear
     writeio(stdout,(7,"  ",20,"  ",(2,6),(3,20.8)),i,countryi[i],ix,j,sharei[i,j],
        countriespopproj[ix,j],popi[i,j],callwait=multpl(j,25),write=false)
   end
   
   for j in endundpyear+1:endyear
     writeio(stdout,(7,"  ",20,"  ",(2,6),(3,20.8)),i,countryi[i],ix,j,sharei[i,endundpyear],
        countriespopproj[ix,j],popi[i,j],callwait=(multpl(j,25)|(j==endyear)),write=false)
   end
   
   countrysums[ix,:] .+= popi[i,:]
   
   worldpop .+= popi[i,:]
   
end        

writeio(stdout,("Done calculating regional populations.",))

writeio(stdout,("Calculating Nordhaus shares...",))

for i in 1:ncells
   yearrange = firstyear:shareyearstart-1
   sharei[i,yearrange] = popi[i,yearrange]./countrysums[countryindexi[i],yearrange]
   writeio(stdout,(6," ",10," ",4,(4,12.8),12.8),i,countryi[i],countryindexi[i],
      sharei[i,1990],sharei[i,1995],sharei[i,2000],sharei[i,2005],sharenordi[i,:],
      callwait=multpl(i,25),write=false)
end   

writeio(stdout,("Done calculating Nordhaus shares.",))

for i in 1:numcountries

    ix = countryindex[i]

    j = startyear - 1
    diff = countrysums[ix,j] - countriespop[ix,4]
    writeio(stdout,((3,6),"  ",20,"  ",(3,20.8)),i,ix,j,countries[ix],countriespop[ix,4],
      countrysums[ix,j],diff,write=false)
      
    for j in startyear:endyear
       diff = countrysums[ix,j] - countriespopproj[ix,j]    
       writeio(stdout,((3,6),"  ",20,"  ",(3,20.8)),i,ix,j,countries[ix],
          countriespopproj[ix,j],countrysums[ix,j],diff,
          callwait=(multpl(j,25)|(j==endyear)),write=false)
    end  
      
end      

writearrays(stdout,(6,20.8),baseyear:endyear,worldpop[baseyear:endyear],nmult=25,
   writeindex=false,write=false)
   
for i in 1:ncells
   if (minimum(popi[i,:]) <= 1e-6)
      writeio(stdout,(8,"  ",20,20.12),i,countryi[i],countriespop[countryindexi[i],:],
         write=false)   
      writeio(stdout,(8,"  ",20,20.12),i,countryi[i],popregi[i,:],write=false)
      writeio(stdout,(8,"  ",20,20.12),i,countryi[i],sharenordi[i,:],callwait=true,
         write=false)
      for j in baseyear:startyear-1
         writeio(stdout,(8,"  ",20,"  ",6,25.12),i,countryi[i],j,popi[i,j],
            callwait=multpl(j,25),write=false)
      end
      for j in startyear:endundpyear
         writeio(stdout,(8,"  ",20,"  ",6,(2,25.12)),i,countryi[i],j,popi[i,j],sharei[i,j],
            callwait=multpl(j,25),write=false)
      end
      for j in endundpyear+1:endyear
         writeio(stdout,(8,"  ",20,"  ",6,(2,25.12)),i,countryi[i],j,popi[i,j],
            sharei[i,endundpyear],callwait=(multpl(j,25)|(j==endyear)),write=false)
      end
   end
end         

writeio(stdout,((2,20.12),),minimum(popi),maximum(popi))

gratei = createoffsetarray(0.0,(1:ncells,baseyear+1:endyear))
for i in 1:ncells
   for j in baseyear+1:endyear
      gratei[i,j] = popi[i,j]/popi[i,j-1] - 1
#      if ((gratei[i,j] > 0.1) | (gratei[i,j] < -0.1))
#         writeio(stdout,((2,6)," ",10," ",(3,13.6),13.6),i,j,countryi[i],gratei[i,j],
#            popi[i,j],popi[i,j-1],popregi[i,:],callwait=true)
#      end      
   end   
end   

writeio(stdout,((2,20.12),),minimum(gratei),maximum(gratei))
writeio(stdout,((2,20.12),),minimum(sharei),maximum(sharei))

fmtstring = "%6.0f%5.0f%5.0f"
fmt = Printf.Format(fmtstring)
open("regpop4.pop","w") do io
   for i in 1:ncells
      writeio(stdout,10,i,write=false)
      Printf.format(io,fmt,i,lati[i],loni[i])
      for j in baseyear:endyear
         @printf(io,"%13.6f",popi[i,j])
      end
      println(io)  
  end
end   
writeio(stdout,("Done writing to regpop4.pop.",))

open("regpop4.grate","w") do io
   for i in 1:ncells
      writeio(stdout,10,i,write=false)
      Printf.format(io,fmt,i,lati[i],loni[i])
      for j in baseyear+1:endyear
         @printf(io,"%10.6f",gratei[i,j])
      end
      println(io)  
  end
end   
writeio(stdout,("Done writing to regpop4.grate.",))

open("regpop4.share","w") do io
   for i in 1:ncells
      writeio(stdout,10,i,write=false)
      Printf.format(io,fmt,i,lati[i],loni[i])
      for j in baseyear:shareyearend
         @printf(io,"%9.6f",sharei[i,j])
      end
      println(io)  
  end
end   
writeio(stdout,("Done writing to regpop4.share.",))


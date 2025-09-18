using DelimitedFiles

include("c:/econ417julia/io4.jl")
include("c:/julia/sort1.jl")
include("c:/climate/createarray1.jl")
include("c:/climate/stats1.jl")

const missingvalue = -99.99
const missingintvalue = -99
const ncells = 20249
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

filename = "c:/Dropbox/noresm/population_growth/shares_method/nord40_gpw_populations.csv"
datamatrix = readdata(filename)
popregi = fill(0.0,ncells,4)
numzero = 0
numlessthanone = 0
nkeep = fill(1,ncells)
for i in 1:ncells
   global numzero,numlessthanone
   popregi[i,1:4] = datamatrix[i+1,5:8]
   if (popregi[i,4] < 1e-6) 
      numzero += 1 
      writeio(stdout,(8,20.12),i,popregi[i,:],callwait=true,write=false)
      popregi[i,4] = popregi[i,1]
   end   
   for j in 2:3
      if (popregi[i,j] < 1e-6)
         popregi[i,j] = popregi[i,1]
      end
   end      
   if (minimum(popregi[i,:]) < 0.001)
      nkeep[i] = 0
      numlessthanone += 1
      writeio(stdout,(8,20.12),i,popregi[i,:],callwait=true,write=false)
   end   
end   

open("regpop3.keep","w") do io
   writearrays(io,(7,5),nkeep)
end   

writeio(stdout,("sum of nkeep: ",10),sum(nkeep))

open("regpop3.popkeep","w") do io
   ix = 0
   for i in 1:ncells
      if (nkeep[i] == 1) 
         ix += 1
         writeio(io,(7,20.8),ix,popregi[i,:])
      end
   end      
end   

writeio(stdout,((2,10),(4,20.12)),numzero,numlessthanone,
   minimum(popregi[:,1]),minimum(popregi[:,2]),
   minimum(popregi[:,3]),minimum(popregi[:,4]),callwait=false,write=true)

writeio(stdout,((2,10),(4,20.6)),numzero,numlessthanone,sum(popregi[:,1]),
   sum(popregi[:,2]),sum(popregi[:,3]),sum(popregi[:,4]),callwait=false,write=true)
   
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
open("regpop3.undp","w") do io
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

wait("Done writing to regpop3.undp.")

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
filename = "c:/Dropbox/noresm/population_growth/nordhaus_update/parse2.gin5"
io = open(filename,"r")
io2 = open("parse2.gin6","w")

icount = 0
for i in 1:ncells

   global icount

   d,lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
      gdpregi1990[i],gdpperregi1990[i] = readio(io,(3,"b3","a40",6))
      
#   if (popregi1990[i] < 0.0005) 
#      writeio(stdout,((3,6),"  ",10,"  ",(3,15.8),13.6),i,lati[i],loni[i],countryi[i],
#         popregi1990[i],gdpregi1990[i],gdpperregi1990[i],popregi[i,:],
#         callwait=true,write=true)
#   end

   if (nkeep[i] == 1)
      icount += 1
      writeio(io2,(6,(2,9)," "^3,40,15.8,12.6,9.4,(3,16.6)),icount,lati[i],loni[i],
         countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
         gdpregi1990[i],gdpperregi1990[i])
   end  

end

close(io)
close(io2)

wait("Done reading parse2.gin5 and writing to parse2.gin6.")


using Formatting

number = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
positiveinteger = r"^[1-9]+[0-9]*$"
beginnumber = r"[+-.]|[0-9]"

function multpl(i,m)

   if (m == 0)
      isamult = false
   else   

      toler = 1.0e-10
   
      ioverm = i/m

      remain = ioverm - trunc(ioverm)
   
      isamult = (remain <= toler)
      
   end
   
   return isamult   
   
end   

function writes(io,s;fieldlength=length(s),numdecimal=0,cr=false)

   blank = " "
   star = "*"

   if isa(s,String)
      lengths = length(s)
      blanklength = max(0,fieldlength-lengths)
      if (blanklength > 0) 
         prints = s[1:lengths] * (blank^blanklength)
      else
         prints = s[1:fieldlength]
      end      
   elseif isa(s,Int64) 
      sstring = string(s)
      lengths = length(sstring)
      if (lengths > fieldlength)
         prints = star^fieldlength
      else
         numpad = fieldlength - lengths
         prints = (blank^numpad) * sstring
      end   
   elseif isa(s,Float64) 
      sstring = format(s,precision=numdecimal)
      lengths = length(sstring)
      if (lengths > fieldlength)
         prints = star^fieldlength
      else
         numpad = fieldlength - lengths
         prints = (blank^numpad) * sstring
      end   
   end   
          
   if cr
      println(io,prints)
   else
      print(io,prints)
   end           

end   

function wait(message="";printwaiting=true,condition=true)
   if condition
      if (length(message) > 0) 
         println(message) 
      end
      if printwaiting 
         print("Waiting...") 
      end
      readline()
   end   
   nothing
end   

function writeio(io,formattuple,varlist...;cr=true,callwait=false,printwaiting=true,write=true,
                 addzero=false)

   if !write 
      return 
   end
   
   if (length(varlist) == 0)

      for i in eachindex(formattuple)
         if isa(formattuple[i],String)
            print(io,formattuple[i])
         else
            wait("Not a string in formattuple.")
         end
      end
               
   else
     
      numformat = 0
      for i in eachindex(formattuple)
         if isa(formattuple[i],Tuple)
            numformat = numformat + formattuple[i][1]
         else   
            numformat += 1
         end
      end         
         
      numvars = length(varlist)
   
      nstring = 0
      for i in eachindex(formattuple)
         if isa(formattuple[i],String) nstring += 1 end
      end      
      
      if ((numformat-nstring) != numvars)
         wait("Argument lists do not match in writeio.")
      end   
      
      ix = 0
      
      for i in eachindex(formattuple)
         if isa(formattuple[i],String)
            print(io,formattuple[i])
         elseif isa(formattuple[i],Tuple)
            if (length(formattuple[i]) != 2) 
               wait("Tuple not the right length in writeio.")
            else
               nrepeat = formattuple[i][1]
               format = formattuple[i][2]
               for i in 1:nrepeat
                  ix += 1
                  if isa(varlist[ix],Int64)
                     if isa(format,Int64)
                        writes(io,varlist[ix],fieldlength=format)
                     else 
                        wait("Format for integer not an integer in repeat loop in writeio.")
                     end
                  elseif isa(varlist[ix],String)
                     if isa(format,Int64)
                        writes(io,varlist[ix],fieldlength=format)
                     else 
                        wait("Format for string not an integer in repeat loop in writeio.")
                     end
                  elseif isa(varlist[ix],Array{Int64})
                     if isa(varlist[ix],Array{Int64,1})
                        if isa(format,Int64)
                           for j in 1:length(varlist[ix])
                              writes(io,varlist[ix][j],fieldlength=format)
                           end   
                        else 
                           wait("Format for integer not an integer in repeat loop in writeio.")
                        end
                     else   
                        wait("Integer array is not a vector in writeio.")
                     end   
                  elseif isa(varlist[ix],Array{String})
                     if isa(varlist[ix],Array{String,1})
                        if isa(format,Int64)
                           for j in 1:length(varlist[ix])
                              writes(io,varlist[ix][j],fieldlength=format)
                           end   
                        else 
                           wait("Format for string not an integer in repeat loop in writeio.")
                        end
                     else   
                        wait("String array is not a vector in writeio.")
                     end   
                  elseif isa(varlist[ix],Float64)
                     if isa(format,Float64)
                        formatstring = string(format)
                        ndecimal = findfirst('.',formatstring)
                        n1 = parse(Int64,formatstring[1:ndecimal-1])
                        n2 = parse(Int64,formatstring[ndecimal+1:length(formatstring)])
                        if addzero 
                           n2 = 10*n2 
                        end      
                        writes(io,varlist[ix],fieldlength=n1,numdecimal=n2)
                     else 
                        wait("Format for real not a real in repeat loop in writeio.")
                     end
                  elseif isa(varlist[ix],Array{Float64})
                     if isa(varlist[ix],Array{Float64,1})
                        if isa(format,Float64)
                           formatstring = string(format)
                           ndecimal = findfirst('.',formatstring)
                           n1 = parse(Int64,formatstring[1:ndecimal-1])
                           n2 = parse(Int64,formatstring[ndecimal+1:length(formatstring)])
                           if addzero 
                              n2 = 10*n2 
                           end      
                           for j in 1:length(varlist[ix])
                              writes(io,varlist[ix][j],fieldlength=n1,numdecimal=n2)
                           end   
                        else 
                           wait("Format for real not a real in repeat loop in writeio.")
                        end
                     else 
                        wait("Real array is not a vector in writeio.")
                     end
                  else
                     wait("Variable not an integer or real in repeat loop in writeio.")
                  end   
               end   
            end    
         else
            ix += 1
            if isa(varlist[ix],String)
               if isa(formattuple[i],Int64)
                  writes(io,varlist[ix],fieldlength=formattuple[i])
               else 
                  wait("Format for string not an integer in writeio.")
               end
            elseif isa(varlist[ix],Int64)
               if isa(formattuple[i],Int64)
                  writes(io,varlist[ix],fieldlength=formattuple[i])
               else 
                  wait("Format for integer not an integer in writeio.")
               end
            elseif isa(varlist[ix],Float64)
               if isa(formattuple[i],Float64)
                  formatstring = string(formattuple[i])
                  ndecimal = findfirst('.',formatstring)
                  n1 = parse(Int64,formatstring[1:ndecimal-1])
                  n2 = parse(Int64,formatstring[ndecimal+1:length(formatstring)])
                  if addzero 
                     n2 = 10*n2 
                  end      
                  writes(io,varlist[ix],fieldlength=n1,numdecimal=n2)
               else 
                  wait("Format for real not a real in writeio.")
               end
            elseif isa(varlist[ix],Array{Int64})
               if isa(varlist[ix],Array{Int64,1})
                  if isa(formattuple[i],Int64)
                     for j in eachindex(varlist[ix])
                        writes(io,varlist[ix][j],fieldlength=formattuple[i])
                     end   
                  else 
                     wait("Format for integer array not an integer in writeio.")
                  end
               else   
                  wait("Integer array is not a vector in writeio.")
               end   
            elseif isa(varlist[ix],Array{String})
               if isa(varlist[ix],Array{String,1})
                  if isa(formattuple[i],Int64)
                     for j in eachindex(varlist[ix])
                        writes(io,varlist[ix][j],fieldlength=formattuple[i])
                     end   
                  else 
                     wait("Format for string array not an integer in writeio.")
                  end
               else   
                  wait("String array is not a vector in writeio.")
               end   
            elseif isa(varlist[ix],Array{Float64})
               if isa(varlist[ix],Array{Float64,1})
                  if isa(formattuple[i],Float64)
                     formatstring = string(formattuple[i])
                     ndecimal = findfirst('.',formatstring)
                     n1 = parse(Int64,formatstring[1:ndecimal-1])
                     n2 = parse(Int64,formatstring[ndecimal+1:length(formatstring)])
                     if addzero 
                        n2 = 10*n2 
                     end      
                     for j in eachindex(varlist[ix])
                         writes(io,varlist[ix][j],fieldlength=n1,numdecimal=n2)
                     end
                  else       
                     wait("Format for real array not a real in writeio.")
                  end    
               else 
                  wait("Real array is not a vector in writeio.")
               end
            else
               wait("Unrecognized type in writeio.")               
            end   
         end
      end  
      
   end   
   
   if cr 
      println(io) 
   end
   
   wait(printwaiting=printwaiting,condition=callwait) 
                   
end

function isanumber(s)

   isanumber = occursin(number,s)
   
end   

function isapositiveinteger(s)

   isapositiveinteger = occursin(positiveinteger,s)
   
end   

function getnumber(s,first,last)

   firstchar = findfirst(beginnumber,s[first:last])

   if (typeof(firstchar) == Nothing) 
      wait("No number in getnumber.") 
   end
   
   startindex = firstchar[1] + first - 1
   blankindex = findfirst(' ',s[startindex:last])
   if typeof(blankindex) == Nothing 
      lastindex = last 
   else   
      lastindex = startindex + blankindex - 2
   end
   
   x = parsenumber(s[startindex:lastindex])
   
   getnumber = (x,lastindex)
   
end    

function getnumber2(s)

   first = 1
   last = length(s)

   firstchar = findfirst(beginnumber,s[first:last])

   if (typeof(firstchar) == Nothing) 
      wait("No number in getnumber.") 
   end
   
   startindex = firstchar[1] + first - 1
   blankindex = findfirst(' ',s[startindex:last])
   if typeof(blankindex) == Nothing 
      lastindex = last 
   else   
      lastindex = startindex + blankindex - 2
   end
   
   x = parsenumber(s[startindex:lastindex])
   
   getnumber2 = x
   
end    

function parsenumber(s)

   if isanumber(s)
      if occursin('.',s)
         x = parse(Float64,s)
      else
         x = parse(Int64,s)
      end
   else
      println(s)
      wait("Not a number in parsenumber.")
   end    

   parsenumber = x       
   
end   

function readio(io,formattuple)

   lengthtuple = length(formattuple)
   
   s = readline(io)
   lengths = length(s)
   t = ()

   if (io==stdin)
      while (lengths == 0)
         s = readline(io)
         lengths = length(s)
      end
   end      
   
   first = 1 
   last = length(s)
   
   for i in eachindex(formattuple)
   
      if isa(formattuple[i],Int64)
      
         for j in 1:formattuple[i]
            (x,lastindex) = getnumber(s,first,last)
             t = (t...,x)   
             first = lastindex + 1
         end
         
      elseif isa(formattuple[i],String)
         
         if (formattuple[i][1] == 'a')
                  
            lengthi = length(formattuple[i])
            
            if isapositiveinteger(formattuple[i][2:lengthi])
               n = parse(Int64,formattuple[i][2:lengthi])
               t = (t...,s[first:min(first+n-1,length(s))])
               first = first + n
            else
               wait("Not a positive integer in a format in readio.")
            end   
            
         elseif (formattuple[i][1] == 'b')   

            lengthi = length(formattuple[i])
            
            if isapositiveinteger(formattuple[i][2:lengthi])
               n = parse(Int64,formattuple[i][2:lengthi])
               first = first + n
            else
               wait("Not an integer in b format in readio.")
            end   
         
         elseif (formattuple[i][1] == 'r')   

            lengthi = length(formattuple[i])
            
            if isapositiveinteger(formattuple[i][2:lengthi])

               n = parse(Int64,formattuple[i][2:lengthi])
               
               (x,lastindex) = getnumber(s,first,last)
               
               a = Array{typeof(x)}(undef,n)
               a[1] = x
               
               first = lastindex + 1
               
               for j in 2:n
                  (x,lastindex) = getnumber(s,first,last)
                  a[j] = x
                  first = lastindex + 1
               end   
               
               t = (t...,a)

            else
               wait("Not an integer in r format in readio.")
            end   
         
         else
            wait("Invalid leading character in format in readio.")
         end
         
      end
      
   end      
   
   if (length(t) == 1)
      readio = t[1]   
   else
      readio = t
   end   
   
end   

# A function for writing a set of equally-sized arrays, with or without a leading index,
# with formatting instructions contained in format.

function writearrays(io,format,arraylist...;nmult=0,writeindex=true,writealllines=true,
                     addzero=false,write=true,lastwait=false)

   if !write 
      return 
   end

   if writealllines
      nvars = length(arraylist)
   else
      nvars = length(arraylist) - 1
   end      

   arraylength = 0
   
   for j in 1:nvars
      if (typeof(arraylist[j]) !== String)
         arraylength = size(arraylist[j],1)     
         break
      end
   end      
   
   if (arraylength == 0)
      wait("No arrays in writearrays.")
   end   

   if writealllines
      writelines = convert.(Bool,ones(Int64,arraylength))     
   else
      writelines = arraylist[length(arraylist)]
   end      
   
   nlineswritten = 0
   
   if writeindex
       
      if !isa(format[1],Int64)
         wait("First element of format is not an integer in writearrays.")
      else
         format2 = (format[1],)
      end   
   
      for i in 2:length(format)
         if isa(format[i],Tuple)
            if (length(format[i]) != 2)
               wait("Invalid format in writearrays.")
            else
               for j in 1:format[i][1]
                  format2 = (format2...,format[i][2])
               end
            end      
         else
            format2 = (format2...,format[i])
         end
      end    
   
      if (length(format2) != nvars+1)
        wait("Format length not correct in writearrays.")
      end   
      
      for i in 1:arraylength

         if writelines[i] 
         
            nlineswritten += 1
          
            writeio(io,format2[1],i,cr=false,addzero=addzero)
       
            for j = 1:nvars
            
               if (typeof(arraylist[j]) == String)
                  writeio(io,format2[j+1],arraylist[j],cr=(j==nvars),addzero=addzero)
               else   
            
                  isamatrix = length(size(arraylist[j])) == 2
            
                  if isamatrix 
                     writeio(io,format2[j+1],arraylist[j][i,:],cr=(j==nvars),addzero=addzero)
                   else   
                     writeio(io,format2[j+1],arraylist[j][i],cr=(j==nvars),addzero=addzero)
                  end   
                  
               end   
         
            end
         
            if multpl(nlineswritten,nmult) 
               wait() 
            end
         
         end
         
      end   
      
   else   
   
      format2 = ()
    
      for i in 1:length(format)
         if isa(format[i],Tuple)
            if (length(format[i]) != 2)
               wait("Invalid format in writearrays.")
            else
               for j in 1:format[i][1]
                  format2 = (format2...,format[i][2])
               end
            end      
         else
            format2 = (format2...,format[i])
         end
      end    
   
      if (length(format2) != nvars)
        wait("Format length not correct in writearrays.")
      end   

      for i in 1:arraylength

         if writelines[i]

            nlineswritten += 1
      
            for j = 1:nvars
            
               if (typeof(arraylist[j]) == String)
                  writeio(io,format2[j],arraylist[j],cr=(j==nvars),addzero=addzero)
               else   
         
                  isamatrix = length(size(arraylist[j])) == 2
            
                 if isamatrix 
                     writeio(io,format2[j],arraylist[j][i,:],cr=(j==nvars),addzero=addzero) 
                  else   
                     writeio(io,format2[j],arraylist[j][i],cr=(j==nvars),addzero=addzero) 
                  end   
                  
               end   
         
            end
         
            if multpl(nlineswritten,nmult) 
               wait() 
            end
       
         end
      
      end   
      
   end   
   
   if lastwait 
      wait()
   end   
   
end           

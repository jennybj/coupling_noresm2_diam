function sort(list;increasing=true)

   nsort = length(list)

   index = collect(1:1:nsort)

   n = nsort
   njump = trunc(Int64,nsort/2)
   while njump >= 1
      ndone = true
      for m in 1:n-njump
         n = m + njump
         if increasing
            if list[index[m]] > list[index[n]]
               ndone = false
               ntemp = index[m]
               index[m] = index[n]
               index[n] = ntemp
            end
         else   
            if list[index[m]] < list[index[n]]
               ndone = false
               ntemp = index[m]
               index[m] = index[n]
               index[n] = ntemp
            end
         end   
      end
      if ndone
         njump = trunc(Int64,njump/2)
      end
   end

   return index

end

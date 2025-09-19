function mean(x;missingvalue=-99.99)

   n = length(x)

   nmissing = sum(missing.(x))

   if (nmissing < n)
      avg = sum(x.*(.!missing.(x)))/(n-nmissing)
   else
      avg = missingvalue
   end

   return avg,nmissing

end

function weightedmean(x,weights;missingvalue=-99.99)

   n = length(x)

   nmissing = sum(missing.(x))

   if (nmissing < n)
      weightedavg = sum((weights.*x) .* (.!missing.(x)))
   else
      weightedavg = missingvalue
   end

   return weightedavg

end

function stddev(x;missingvalue=-99.99)

   n = length(x)

   avg,nmissing = mean(x)

   if (nmissing < n)
      diff = x .- avg
      sd = sqrt(sum(diff.*diff.*(.!missing(x)))/(n-nmissing))
   else
      sd = missingvalue
   end

   return sd

end

function findrange(x;missingvalue=-99.99)

   t = ()

   for i in 1:length(x)
      if !missing(x[i])
         t = (t...,x[i])
      end
   end

   if length(t) == 0
      max = missingvalue
      min = missingvalue
   else
      max = maximum(t)
      min = minimum(t)
   end

   return max,min

end

function median(x;missingvalue=-99.99)

   t = ()

   for i in 1:length(x)
      if !missing(x[i])
         t = (t...,x[i])
      end
   end

   n = length(t)

   if n == 0
      med = missingvalue
   else
      index = sort(t)
      ix = trunc(Int64,n/2)
      if multpl(n,2)
         med = (t[index[ix]] + t[index[ix+1]])/2.0
      else
         med = t[index[ix]]
      end
   end

   return med

end

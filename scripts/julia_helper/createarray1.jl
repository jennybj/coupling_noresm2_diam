#x = MArray{Tuple{3,2,4},Float64}(undef)
#z = MMatrix{3,3,Float64}(undef)

using OffsetArrays

function createarray(s,dimlist...)

   ndim = length(dimlist)

   a = Array{typeof(s),length(dimlist)}(undef,dimlist)

   a .= s

   return a

end

function createoffsetarray(s,ranges)

   n = length(ranges)
   
   dim = (length(ranges[1]),)
   
   for i in 2:n
      dim = (dim...,length(ranges[i]))
   end   
   
   a = OffsetArray(fill(s,dim),ranges)
   
   return a
   
end   


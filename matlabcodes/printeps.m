classdef printeps < handle
    properties
        count
        countref
        epsilon
    end
    methods 
        function out=printout(obj)
           if obj.count==obj.countref,
               filename=['epsitr',num2str(obj.count),'.h5'];
               hdf5write(filename,'/eps',obj.epsilon);
               out=1;
           else
               out=0;
           end
        end
    end

end
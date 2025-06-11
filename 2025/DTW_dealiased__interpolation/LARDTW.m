
 [Distance Path] = LARDTW(signals, reference, pathConstraintPer, regionWidthPer, ...
            MinScale, MaxScale);
        pathConstraint = ceil(pathConstraintPer * length);
        regionWidth = ceil(regionWidthPer * length);
        DTWDistMatrix = 1000000*ones(length,length);
        DTWDirMatrix = -1*ones(length,length);
        DTWLenMatrix = -1*ones(length,length);
        fullConstraintWidth = 1 + 2*pathConstraint;
        regionBuffer = -1*ones(2,fullConstraintWidth,10);
        for i=1:length
            if mod(i,2)==0
                regionStoreIndex = 0;
			    regionLoadIndex = 1;
            else
                regionStoreIndex = 1;
			    regionLoadIndex = 0;
                regionLoadBuffer = regionBuffer(regionLoadIndex);
                regionStoreBuffer = regionBuffer(regionStoreIndex);
            end
                for j=max(i-pathConstraint,0):min(i+pathConstraint,length-1)
                    bufferIndex = j - i + pathConstraint;
                    mStart = -regionWidth;
                    mEnd = regionWidth;
                    if (i - regionWidth) < 0 || (j - regionWidth) < 0
                        mStart = mStart-min(i - regionWidth, j - regionWidth);
                    elseif (i + regionWidth) >= length || (j + regionWidth) >= length
                        mEnd =  mEnd - max(i + regionWidth - length + 1, j + regionWidth - length + 1);
                        w = -mStart + mEnd + 1;
                        point_distance = 0;
                        rho = 0;
                        gamma = 0;
                        tau = 0;
                        phi = 0;
                        eta = 0;
                    elseif i == 0 || j == 0
                        for m = mStart:mEnd
                            iInd = i+m;
                            jInd = j+m;
                            rho = rho + signal(iInd)*reference(jInd);
                            gamma = gamma + reference(jInd)*reference(jInd);
                            tau = tau + reference(jInd);
                            phi = phi + signal(iInd);
					        eta = eta + signal(iInd)*signal(iInd);
                        end
                    else
                        iInd = i+mEnd;
                        jInd = j+mEnd;
                        if mEnd < regionWidth
                            rho = -regionLoadBuffer(bufferIndex)(1) + regionLoadBuffer(bufferIndex)(0);
                            gamma = -regionLoadBuffer(bufferIndex)(3) + regionLoadBuffer(bufferIndex)(2);
					        tau = -regionLoadBuffer(bufferIndex)(5) + regionLoadBuffer(bufferIndex)(4);
					        phi = -regionLoadBuffer(bufferIndex)(7) + regionLoadBuffer(bufferIndex)(6);
					        eta = -regionLoadBuffer(bufferIndex)(9) + regionLoadBuffer(bufferIndex)(8);
                        else
                            rho = -regionLoadBuffer(bufferIndex)(1) + regionLoadBuffer(bufferIndex)(0) + signal(iInd)*reference(jInd);
					        gamma = -regionLoadBuffer(bufferIndex)(3) + regionLoadBuffer(bufferIndex)(2) + reference(jInd)*reference(jInd);
					        tau = -regionLoadBuffer(bufferIndex)(5) + regionLoadBuffer(bufferIndex)(4) + reference(jInd);
					        phi = -regionLoadBuffer(bufferIndex)(7) + regionLoadBuffer(bufferIndex)(6) + signal(iInd);
					        eta = -regionLoadBuffer(bufferIndex)(9) + regionLoadBuffer(bufferIndex)(8) + signal(iInd)*signal(iInd);
                        end
                    end
                    regionStoreBuffer(bufferIndex)(0) = rho;
		         	regionStoreBuffer(bufferIndex)(2) = gamma;
                    regionStoreBuffer(bufferIndex)(4) = tau;
                    regionStoreBuffer(bufferIndex)(6) = phi;
                    regionStoreBuffer(bufferIndex)(8) = eta;
                    
                    if mStart > -regionWidth
                        regionStoreBuffer(bufferIndex)(1) = 0;
                        regionStoreBuffer(bufferIndex)(3) = 0;
                        regionStoreBuffer(bufferIndex)(5) = 0;
                        regionStoreBuffer(bufferIndex)(7) = 0;
                        regionStoreBuffer(bufferIndex)(9) = 0;
                    else
                        iInd = i+mStart;
                        jInd = j+mStart;
                        regionStoreBuffer(bufferIndex)(1) = signal(iInd)*reference(jInd);
                        regionStoreBuffer(bufferIndex)(3) = reference(jInd)*reference(jInd);
                        regionStoreBuffer(bufferIndex)(5) = reference(jInd);
                        regionStoreBuffer(bufferIndex)(7) = signal(iInd);
                        regionStoreBuffer(bufferIndex)(9) = signal(iInd)*signal(iInd);
                    end
                    value = (gamma - 1/w*tau*tau);
                    if value == 0
                        value = 0.0000001;
                    end
                    c = (rho - 1/w*phi*tau)/value; 
                    if c > cMax
                        c = cMax;
                    elseif c < cMin
                        c = cMin;
                    end
                    e = 1/w * (phi - c*tau);
                    point_distance = 1/w * (eta - 2*c*rho - 2*e*phi + c*c*gamma + 2*c*e*tau + w*e*e);
                    
                    
                    if j-1>=0
                        horizontal_distance = DTWDistMatrix(i)(j-1);
                    else
                         horizontal_distance = inf;
                    end
                    if i-1 >= 0
                        vertical_distance = DTWDistMatrix[i-1][j];
                    else
                        vertical_distance = inf;
                    end
                    if i-1 >= 0 && j-1 >= 0
                        diagonal_distance = DTWDistMatrix[i-1][j-1];
                    else
                         diagonal_distance = inf;
                    end
                    if j-1 >= 0
                        horizontal_len = DTWLenMatrix[i][j-1];
                    else
                         horizontal_len = inf;
                    end
                    if i-1 >= 0
                        vertical_len = DTWLenMatrix[i-1][j];
                    else
                        vertical_len = inf;
                    end
                    if i-1 >= 0 && j-1 >= 0
                        diagonal_len = DTWLenMatrix[i-1][j-1]
                    else
                        diagonal_len = inf;
                    end
                    direction = -1;
                    prev_distance = 0;
                    prev_len = 0;
                    diagonal_distance = diagonal_distance - eps;  %%EPS定义
                    if i ~= 0 || j ~= 0
                        if diagonal_distance <= vertical_distance && diagonal_distance <= horizontal_distance
                            direction = DIAGONAL;
                            prev_distance = diagonal_distance;
                            prev_len = diagonal_len;
                        elseif vertical_distance < diagonal_distance && vertical_distance <= horizontal_distance
                            direction = VERTICAL;
                            prev_distance = vertical_distance;
                            prev_len = vertical_len;
                        else
                            direction = HORIZONTAL;
                            prev_distance = horizontal_distance;
                            prev_len = horizontal_len;
                        end
                    end
                    DTWDistMatrix[i][j] = point_distance + prev_distance;
                    DTWDirMatrix[i][j] = direction;
                    DTWLenMatrix[i][j] = 1 + prev_len;
                end
        end
        distance = DTWDistMatrix[length-1][length-1];
        pathLen = DTWLenMatrix[length-1][length-1];
        % traverse DTW direction matrix to obtain the warp path
        
	   % warpedPath = (int**) malloc((*pathLen) * sizeof(int*)); %%定义需改
        for m=1:pathLen
           % warpedPath[m] = (int*) malloc(2 * sizeof(int)); %%
            warpedPath[m][0] = 0;
            warpedPath[m][1] = 0;
        end
                            
                        
                    
                    
                        
                    
                    
                        
                            
                            
                            
                        
                        
                        
                        
                    
                
            
        
        
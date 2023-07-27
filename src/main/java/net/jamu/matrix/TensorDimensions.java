/*
 * Copyright 2023 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

/**
 * Basic tensor properties mostly related to the dimension of a tensor.
 * 
 * @since 1.4.0
 */
public interface TensorDimensions extends Dimensions {

    /**
     * Returns the depth (&gt;= 1) (i.e., the number of matrices) of this
     * tensor.
     * 
     * @return the depth of this tensor
     */
    public int numDepth();

    /**
     * Returns the offset (in number of array elements) from the start of one
     * matrix to the start of the next matrix.
     * 
     * @return the offset to the next matrix
     */
    public int stride();
}

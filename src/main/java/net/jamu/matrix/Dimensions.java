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
 * Basic properties of a matrix or a tensor related to its dimensions.
 */
public interface Dimensions {

    /**
     * Returns the number of columns (&gt;= 1) of this matrix or tensor.
     * 
     * @return the number of matrix or tensor columns
     */
    int numColumns();

    /**
     * Returns the number of rows (&gt;= 1) of this matrix or tensor.
     * 
     * @return the number of matrix or tensor rows
     */
    int numRows();
}

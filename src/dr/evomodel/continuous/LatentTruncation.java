/*
 * LatentTruncation.java
 *
 * Copyright © 2002-2024 the BEAST Development Team
 * http://beast.community/about
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 *
 */

package dr.evomodel.continuous;

import dr.math.distributions.Distribution;

/**
 * @author Marc A. Suchard
 */
public interface LatentTruncation {

    boolean validTraitForTip(int tip);

    double getNormalizationConstant(Distribution working);

    double getLogLikelihood();

    public abstract class Delegate {

        public double getNormalizationConstant(Distribution working) {
            if (!normalizationKnown) {
                normalizationConstant = computeNormalizationConstant(working);
                normalizationKnown = true;
            }
            return normalizationConstant;
        }

        public void setNormalizationKnown(boolean value) {
            normalizationKnown = value;
        }

        protected abstract double computeNormalizationConstant(Distribution working);

        private boolean normalizationKnown = false;
        private double normalizationConstant;
    }
}

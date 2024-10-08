/*
 * ASRSubstitutionModelConvolutionStatisticParser.java
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

package dr.evomodelxml.treedatalikelihood;

import dr.evolution.tree.TreeUtils;
import dr.evolution.util.Taxa;
import dr.evolution.util.TaxonList;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.substmodel.SubstitutionModel;
import dr.evomodel.treedatalikelihood.discrete.ASRSubstitutionModelConvolutionStatistic;
import dr.evomodel.treedatalikelihood.discrete.SequenceDistanceStatistic;
import dr.evomodel.treelikelihood.AncestralStateBeagleTreeLikelihood;
import dr.inference.distribution.ParametricDistributionModel;
import dr.inference.model.Statistic;
import dr.xml.*;

import static dr.evomodelxml.tree.MonophylyStatisticParser.parseTaxonListOrTaxa;

/**
 */
public class ASRSubstitutionModelConvolutionStatisticParser extends AbstractXMLObjectParser {

    public static String STATISTIC = "asrSubstitutionModelConvolutionStatistic";
    public static String SUBS_MODEL_ANCESTOR = "substitutionModelAncestor";
    public static String SUBS_MODEL_DESCENDANT = "substitutionModelDescendant";
    public static String DOUBLETS = "doublets";
    public static String DOUBLETS_TO = "doubletsTo";
    public static String PAIR_SUBS_MODEL_ANCESTOR = "doubletSubstitutionModelAncestor";
    public static String PAIR_SUBS_MODEL_DESCENDANT = "doubletSubstitutionModelDescendant";
    public static String RATE_ANCESTOR = "rateAncestor";
    public static String RATE_DESCENDANT = "rateDescendant";
    private static final String MRCA = "mrca";
    public static final String TAXA = "taxa";
    public static final String BOOT = "bootstrap";
    public static final String PRIOR = "prior";
    public static final String DISTANCE = "takeDistanceAsFixed";
    public static final String PRESENT = "anchorToPresent";

    public String getParserName() { return STATISTIC; }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        String name = xo.getAttribute(Statistic.NAME, xo.getId());

        AncestralStateBeagleTreeLikelihood asrLike = (AncestralStateBeagleTreeLikelihood) xo.getChild(AncestralStateBeagleTreeLikelihood.class);

        SubstitutionModel subsModelAncestor = null;
        if (xo.hasChildNamed(SUBS_MODEL_ANCESTOR)) {
            subsModelAncestor = (SubstitutionModel) xo.getChild(SUBS_MODEL_ANCESTOR).getChild(0);
        }

        SubstitutionModel subsModelDescendant = null;
        if (xo.hasChildNamed(SUBS_MODEL_DESCENDANT)) {
            subsModelDescendant = (SubstitutionModel) xo.getChild(SUBS_MODEL_DESCENDANT).getChild(0);
        }

        int[] doublets = new int[0];
        if ( xo.hasAttribute(DOUBLETS) ) {
            doublets = xo.getIntegerArrayAttribute(DOUBLETS);
        }

        int[] doubletsTo = new int[0];
        if ( xo.hasAttribute(DOUBLETS_TO) ) {
            doubletsTo = xo.getIntegerArrayAttribute(DOUBLETS_TO);
        }

        SubstitutionModel pairedSubsModelAncestor = null;
        if (xo.hasChildNamed(PAIR_SUBS_MODEL_ANCESTOR)) {
            pairedSubsModelAncestor = (SubstitutionModel) xo.getChild(PAIR_SUBS_MODEL_ANCESTOR).getChild(0);
        }

        SubstitutionModel pairedSubsModelDescendant = null;
        if (xo.hasChildNamed(PAIR_SUBS_MODEL_DESCENDANT)) {
            pairedSubsModelDescendant = (SubstitutionModel) xo.getChild(PAIR_SUBS_MODEL_DESCENDANT).getChild(0);
        }

        BranchRateModel branchRates = (BranchRateModel)xo.getChild(BranchRateModel.class);

        boolean boot = xo.getAttribute(BOOT, false);
        boolean takeDistanceAsFixed = xo.getAttribute(DISTANCE, false);
        boolean anchorAtPresent = xo.getAttribute(PRESENT, true);

        TaxonList mrcaTaxa = null;
        if (xo.hasChildNamed(MRCA)) {
            mrcaTaxa = parseTaxonListOrTaxa(xo.getChild(MRCA));
        }

        ParametricDistributionModel prior = null;
        if ( xo.hasChildNamed(PRIOR) ) {
            prior = (ParametricDistributionModel) xo.getElementFirstChild(PRIOR);
        }

        Statistic rateAncestor = null;
        if (xo.hasChildNamed(RATE_ANCESTOR)) {
            rateAncestor = (Statistic) xo.getChild(RATE_ANCESTOR).getChild(0);
            if (rateAncestor.getDimension() != 1) {
                throw new RuntimeException("If providing ancestor rate, it must be a 1-dimensional statistic.");
            }

        }

        Statistic rateDescendant = null;
        if (xo.hasChildNamed(RATE_DESCENDANT)) {
            rateDescendant = (Statistic) xo.getChild(RATE_DESCENDANT).getChild(0);
            if (rateDescendant.getDimension() != 1) {
                throw new RuntimeException("If providing descendent rate, it must be a 1-dimensional statistic.");
            }

        }

//        TaxonList mrcaTaxa = null;
//        if (xo.hasChildNamed(MRCA)) {
//            mrcaTaxa = (TaxonList) xo.getElementFirstChild(MRCA);
//        }

        ASRSubstitutionModelConvolutionStatistic stat = null;
        try {
            stat = new ASRSubstitutionModelConvolutionStatistic(
                    name,
                    asrLike,
                    subsModelAncestor,
                    subsModelDescendant,
                    doublets,
                    doubletsTo,
                    pairedSubsModelAncestor,
                    pairedSubsModelDescendant,
                    branchRates,
                    rateAncestor,
                    rateDescendant,
                    takeDistanceAsFixed,
                    anchorAtPresent,
                    mrcaTaxa,
                    boot,
                    prior);
        } catch (TreeUtils.MissingTaxonException e) {
            throw new XMLParseException("Unable to find taxon-set.");
        }

        return stat;
    }

    //************************************************************************
    // AbstractXMLObjectParser implementation
    //************************************************************************

    public String getParserDescription() {
        return "Estimates (via ML or MAP) branch time prior to MRCA at which substitution regime shifts from ancestor to descendant model.";
    }

    public Class getReturnType() { return SequenceDistanceStatistic.class; }

    public XMLSyntaxRule[] getSyntaxRules() { return rules; }

    private XMLSyntaxRule[] rules = new XMLSyntaxRule[] {
            AttributeRule.newBooleanRule(BOOT, true, "Should the ASR be bootstrapped to account for uncertainty in the MLE?"),
            new ElementRule(AncestralStateBeagleTreeLikelihood.class, false),
            new ElementRule(SUBS_MODEL_ANCESTOR, SubstitutionModel.class, "Substitution model for the ancestral portion of the branch.", false),
            new ElementRule(SUBS_MODEL_DESCENDANT, SubstitutionModel.class, "Substitution model for the more recent portion of the branch.", false),
            AttributeRule.newIntegerArrayRule(DOUBLETS, true), // Integer codes for all doublets as "doublet1.1 doublet1.2 ... doubletn.1 doubletn.2"
            AttributeRule.newIntegerArrayRule(DOUBLETS_TO, true), // Optional codes for specifying to only partition doublets mutating from doublets to doubletsTo
            new ElementRule(PAIR_SUBS_MODEL_ANCESTOR, SubstitutionModel.class, "Optional doublet substitution model for the ancestral portion of the branch (results in a partitioned context-dependent model).", true),
            new ElementRule(PAIR_SUBS_MODEL_DESCENDANT, SubstitutionModel.class, "Optional doublet substitution model for the more recent portion of the branch (results in a partitioned context-dependent model).", true),
            new ElementRule(BranchRateModel.class, false),
            new ElementRule(RATE_ANCESTOR, Statistic.class, "If provided, this will be used as the evolutionary rate for the ancestral portion of the branch instead of the rate provided by the BranchRateModel.", true),
            new ElementRule(RATE_DESCENDANT, Statistic.class, "If provided, this will be used as the evolutionary rate for the descendant portion of the branch instead of the rate provided by the BranchRateModel.", true),
            new ElementRule(MRCA,
                    new XMLSyntaxRule[]{new ElementRule(Taxa.class)}, false),
            new ElementRule(PRIOR, ParametricDistributionModel.class, "A prior for the convolution time (measured in time before descendant node).", true),
            AttributeRule.newBooleanRule(DISTANCE, true, "If true, the distance along the branch (branchRate * branchTime) is taken as fixed, the rates/times may be modified."),
            AttributeRule.newBooleanRule(PRESENT, true, "If using distance, should the descendant (rather than ancestral) rate be taken to be fixed?"),
    };

}

/*
 * CladeSystem.java
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

package dr.app.tools.newtreeannotator;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.util.TaxonList;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * @author Andrew Rambaut
 * @version $
 */
final class CladeSystem {
    private double treeCount = 0;

    /**
     * Constructor starting with an empty clade system
     */
    public CladeSystem() {
    }

    /**
     * Constructor adding a single target tree
     */
    public CladeSystem(Tree targetTree) {
        add(targetTree);
    }
    /**
     * adds all the clades in the tree
     */
    public void add(Tree tree) {
        if (taxonList == null) {
            taxonList = tree;
        }

        if (treeCount == 0) {
            // these will always be the same so create them once
            addTipClades(tree);
        }

        // Recurse over the tree and add all the clades (or increment their
        // frequency if already present). The root clade is added too (for
        // annotation purposes).
        rootClade = addClades(tree, tree.getRoot());

        assert rootClade.getSize() == tree.getExternalNodeCount();

        treeCount += 1;
    }

    public Clade getRootClade() {
        return rootClade;
    }

    /**
     * add all the tips in a tree
     */
    private void addTipClades(Tree tree) {
        for (int i = 0; i < tree.getExternalNodeCount(); i++) {
            NodeRef tip = tree.getExternalNode(i);
            int index = tip.getNumber();
            Clade clade = new BiClade(index, tree.getNodeTaxon(tip));
            tipClades.put(index, clade);
        }
    }
    /**
     * recursively add all the clades in a tree
     */
    private Clade addClades(Tree tree, NodeRef node) {
        Clade clade;
        if (tree.isExternal(node)) {
            // all tip clades should already be there
            clade = tipClades.get(node.getNumber());
            assert clade.getTaxon().equals(tree.getNodeTaxon(node));
        } else {
            assert tree.getChildCount(node) == 2 : "requires a strictly bifurcating tree";

            Clade clade1 = addClades(tree, tree.getChild(node, 0));
            Clade clade2 = addClades(tree, tree.getChild(node, 1));
            clade = getOrAddClade(clade1, clade2);
        }
        assert clade != null;

        clade.setCount(clade.getCount() + 1);

        return clade;
    }

    /**
     * see if a clade exists otherwise create it
     */
    private Clade getOrAddClade(Clade child1, Clade child2) {
        BiClade clade = (BiClade)cladeMap.get(BiClade.makeKey(child1.getKey(), child2.getKey()));
        if (clade == null) {
            clade = new BiClade(child1, child2);
            cladeMap.put(clade.getKey(), clade);
        } else {
            clade.addSubClades(child1, child2);
        }
        return clade;
    }

    public Clade getClade(Object key) {
        if (key instanceof Integer) {
            return tipClades.get(key);
        }
        return cladeMap.get(key);
    }

    public void traverseTree(Tree tree, CladeAction action) {
        traverseTree(tree, tree.getRoot(), action);
    }

    private Object traverseTree(Tree tree, NodeRef node, CladeAction action) {

        Object key;

        if (tree.isExternal(node)) {
            key = node.getNumber();
        } else {
            assert tree.getChildCount(node) == 2;

            Object key1 = traverseTree(tree, tree.getChild(node, 0), action);
            Object key2 = traverseTree(tree, tree.getChild(node, 1), action);

            key = BiClade.makeKey(key1, key2);
        }

        Clade clade = getClade(key);
        if (clade != null) {
            action.actOnClade(clade, tree, node);
        } else {
            assert action.expectAllClades();
        }

        return key;
    }

    public void calculateCladeCredibilities(int totalTreesUsed) {
        for (Clade clade : cladeMap.values()) {
            assert clade.getCount() <= totalTreesUsed : "clade.getCount=(" + clade.getCount() +
                        ") should be <= totalTreesUsed = (" + totalTreesUsed + ")";

            clade.setCredibility(((double) clade.getCount()) / (double) totalTreesUsed);
        }
    }

    public double getLogCladeCredibility(Tree tree) {
        final double[] logCladeCredibility = {0.0};
        traverseTree(tree, new CladeAction() {
            @Override
            public void actOnClade(Clade clade, Tree tree, NodeRef node) {
                logCladeCredibility[0] += Math.log(clade.getCredibility());
            }

            @Override
            public boolean expectAllClades() {
                return true;
            }
        });
        return logCladeCredibility[0];
    }

    public int getCladeCount() {
        return cladeMap.keySet().size();
    }

    //
    // Private stuff
    //
    TaxonList taxonList = null;
    private final Map<Object, Clade> tipClades = new HashMap<>();
   private final Map<Object, Clade> cladeMap = new HashMap<>();

    Clade rootClade;
}

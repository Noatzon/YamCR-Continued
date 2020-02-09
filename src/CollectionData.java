import java.util.ArrayList;
import java.util.Collection;

import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;

import forohfor.scryfall.api.Card;

class CollectionData extends DefaultTableModel {
    private static final long serialVersionUID = 1L;
    private ArrayList<CollectionEntry> data;
    private static final String[] NAMES = {"Name", "Set Code", "Foil", "Count"};

    private boolean showCount = false;

    public CollectionData(boolean showCount)
    {
        data = new ArrayList<CollectionEntry>();
        this.showCount = showCount;
    }

    public CollectionEntry get(int ix)
    {
        return data.get(ix);
    }

    public void addEntry(CollectionEntry ent)
    {
        data.add(ent);
        fireTableDataChanged();
    }

    public void addCard(Card c)
    {
        data.add(new CollectionEntry(c));
        fireTableDataChanged();
    }

    public void addCards(Collection<Card> cards)
    {
        for(Card c: cards)
        {
            addCard(c);
        }
    }

    @Override
    public int getRowCount() {
        if(data != null)
        {
            return data.size();
        }
        return 0;
    }

    @Override
    public int getColumnCount() {
        return (showCount ? NAMES.length:NAMES.length-1);
    }

    @Override
    public String getColumnName(int columnIndex) {
        return NAMES[columnIndex];
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        return String.class;
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return false;
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        CollectionEntry coll = data.get(rowIndex);
        switch(columnIndex)
        {
            case 0:
                return coll.getName();
            case 1:
                return coll.getSetCode();
            case 2:
                return coll.isFoil() ? "✓":"";
            case 3:
                return ""+coll.getCount();
        }
        return "";
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
    }

    @Override
    public void addTableModelListener(TableModelListener l) {

    }

    @Override
    public void removeTableModelListener(TableModelListener l) {

    }
}
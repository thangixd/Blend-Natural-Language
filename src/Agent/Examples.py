from dataclasses import dataclass, field

from src.Agent import Planner
from src.Agent.Plans import Combiner, Part, SearchPlan

from typing import Any, Dict, List, Optional, Tuple, Union
from src.Agent.Plans import TemplateId

TABLES = {
    'Address_SEP_table_45.csv': 'AddressID (numeric), AddressLine1 (text), AddressLine2 (text), City (text), '
                                'StateProvinceID (numeric), PostalCode (text), SpatialLocation (text), rowguid (text), '
                                'ModifiedDate (text)',
    'vEmployee_SEP_table_42.csv': 'BusinessEntityID (numeric), Title (text), FirstName (text), MiddleName (text), '
                                  'LastName (text), Suffix (text), JobTitle (text), PhoneNumber (text), '
                                  'PhoneNumberType (text), EmailAddress (text), EmailPromotion (numeric), '
                                  'AddressLine1 (text), AddressLine2 (text), City (text), StateProvinceName (text), '
                                  'PostalCode (text), CountryRegionName (text), AdditionalContactInfo (numeric)',
    'vStoreWithContacts_SEP_table_26.csv': 'BusinessEntityID (numeric), Name (text), ContactType (text), Title (text), '
                                           'FirstName (text), MiddleName (text), LastName (text), Suffix (text), '
                                           'PhoneNumber (text), PhoneNumberType (text), EmailAddress (text), '
                                           'EmailPromotion (numeric)',
    'SalesTerritory_SEP_table_70.csv': 'TerritoryID (numeric), Name (text), CountryRegionCode (text), Group (text), '
                                       'SalesYTD (numeric), SalesLastYear (numeric), CostYTD (numeric), CostLastYear '
                                       '(numeric), rowguid (text), ModifiedDate (text)',
    'vPersonDemographics_SEP_table_40.csv': 'BusinessEntityID (numeric), TotalPurchaseYTD (numeric), '
                                            'DateFirstPurchase (text), BirthDate (text), MaritalStatus (text), '
                                            'YearlyIncome (text), Gender (text), TotalChildren (numeric), '
                                            'NumberChildrenAtHome (numeric), Education (text), Occupation (text), '
                                            'HomeOwnerFlag (text), NumberCarsOwned (numeric)',
    'JobCandidate_SEP_table_41.csv': 'JobCandidateID (numeric), BusinessEntityID (numeric), Resume (text), '
                                     'ModifiedDate (text)',
    'vProductAndDescription_SEP_table_35.csv': 'ProductID (numeric), Name (text), ProductModel (text), CultureID '
                                               '(text), Description (text)',
}
ADDRESS, EMPLOYEE, CONTACTS, TERRITORIES, DEMOGRAPHICS, CANDIDATES, DESCRIPTIONS = TABLES


@dataclass(frozen=True)
class Example:
    table: str
    request: str
    plan: SearchPlan
    alternatives: Tuple[Part, ...] = ()
    readings: Tuple[Union[Part, Combiner], ...] = ()
    template: Optional[TemplateId] = None
    slots: Dict[str, Any] = field(default_factory=dict)


def _part(part_id: str, words: str, k: Optional[int], seeker: str, /, rows: Tuple = (), **slots: Any) -> Part:
    fill = {'seeker': seeker, **({'rows': rows} if rows else {})}
    fill |= {name: value if isinstance(value, dict) else {'value': value, 'status': 'certain'}
             for name, value in slots.items()}
    return Part(id=part_id, words=words, fill=fill, k=k)


def _plan(root: Union[Part, Combiner], k: int = 10, status: str = 'not_stated') -> SearchPlan:
    return SearchPlan(version=0, k={'value': k, 'status': status}, root=root)


def _in_tree(root: Combiner, template: TemplateId, ids: Tuple[str, ...]) -> SearchPlan:
    return _plan(root).model_copy(update={'template': template, 'template_parts': ids})


def _unnamed(value: Any) -> Dict[str, Any]:
    return {'value': value, 'status': 'not_stated'}


UNNAMED_JOIN = {'value': 'City', 'status': 'not_stated', 'alternatives': ('PostalCode',)}
NAMES = {'value': 'Name', 'status': 'ambiguous', 'alternatives': ('LastName', 'FirstName')}
UNNAMED_COLUMN = {'value': 'EmailAddress', 'status': 'not_stated', 'alternatives': ('PhoneNumber', 'LastName')}
UNNAMED_KEY = _unnamed('BusinessEntityID')
NEGATIVE_ROWS = (('Bothell', '79'), ('Seattle', '79'))
PARIS_JOIN = _part('p1', 'Tables about Paris', 100, 'single_column', rows=(('Paris',),), column='City')

TASK_WORDS = 'Most of my job candidates have no BusinessEntityID yet. Find tables that could supply it'
MORE_ROWS = 'I want more rows like my product descriptions to add to this table'

DIRECTION_KEPT = _part('p1', 'the tables with a column correlating with my SalesLastYear joined on CountryRegionCode',
                       100, 'correlation', join='CountryRegionCode', target='SalesLastYear')
DIRECTION_REMOVED = _part('p2', 'the tables that mention Europe', 1000, 'keyword', words=('Europe',))
REMOVAL_KEPT = _part('p1', 'Tables that join on my City column', 100, 'single_column', column='City')
REMOVAL_REMOVED = _part('p2', 'the tables a search for Redmond turns up', 1000, 'keyword', words=('Redmond',))

EXAMPLES: Dict[str, Example] = {
    'single_column_join': Example(ADDRESS, 'Find tables I can join with my table on City', _plan(
        _part('p1', 'tables I can join with my table on City', 100, 'single_column', column='City')),
        template='single_column_join', slots={'column': 'City'}),
    'multi_column_join': Example(ADDRESS, 'Tables that have my cities together with their postal codes', _plan(
        _part('p1', 'Tables that have my cities together with their postal codes', None, 'multi_column',
              columns=('City', 'PostalCode'))),
        template='multi_column_join', slots={'columns': ('City', 'PostalCode')}),
    'keyword_search': Example(ADDRESS, 'Which tables mention 1970 Napa Ct. or 9833 Mt. Dias Blv.?', _plan(
        _part('p1', 'tables mention 1970 Napa Ct. or 9833 Mt. Dias Blv.', None, 'keyword',
              words=('1970 Napa Ct.', '9833 Mt. Dias Blv.'))), template='keyword',
        slots={'words': ('1970 Napa Ct.', '9833 Mt. Dias Blv.')}),
    'correlation_search': Example(
        ADDRESS, 'Tables with a numeric column that correlates with my StateProvinceID',
        _plan(_part('p1', 'Tables with a numeric column that correlates with my StateProvinceID', 100, 'correlation',
                    join=UNNAMED_JOIN, target='StateProvinceID')),
        template='correlation', slots={'join': UNNAMED_JOIN, 'target': 'StateProvinceID'}),
    'nl_search': Example(ADDRESS, 'Find tables about bike shop customers and where they live', _plan(
        _part('p1', 'tables about bike shop customers and where they live', None, 'natural_language',
              description='bike shop customers and where they live')), template='nl_search',
        slots={'description': 'bike shop customers and where they live'}),
    'typed_rows': Example(ADDRESS, 'Tables with a column that holds the cities Paris or Berlin', _plan(
        _part('p1', 'Tables with a column that holds the cities Paris or Berlin', 100, 'single_column',
              rows=(('Paris',), ('Berlin',)), column='City'))),
    'pairs_and_join': Example(
        ADDRESS, 'Tables that hold my address lines together with their cities and that join on my postal codes',
        _plan(Combiner(id='c1', kind='intersection', inputs=(
            _part('p1', 'Tables that hold my address lines together with their cities', 100, 'multi_column',
                  columns=('AddressLine1', 'City')),
            _part('p2', 'join on my postal codes', 100, 'single_column', column='PostalCode'))))),
    'union_search': Example(ADDRESS, 'Tables that join on my address lines, on my cities or on my postal codes', _plan(
        Combiner(id='c1', kind='union', k=300, inputs=(
            _part('p1', 'join on my address lines', 100, 'single_column', column='AddressLine1'),
            _part('p2', 'on my cities', 100, 'single_column', column='City'),
            _part('p3', 'on my postal codes', 100, 'single_column', column='PostalCode'))))),
    'complex_search': Example(
        ADDRESS, 'The 5 tables that join on most of my columns AddressLine1, City and PostalCode, or that join on my '
                 'address lines, or that have a column correlating with my StateProvinceID joined on AddressLine1',
        _plan(Combiner(id='c1', kind='union', k=250, inputs=(
            Combiner(id='c2', kind='counter', k=150, inputs=(
                _part('p1', 'AddressLine1', 50, 'single_column', column='AddressLine1'),
                _part('p2', 'City', 50, 'single_column', column='City'),
                _part('p3', 'PostalCode', 50, 'single_column', column='PostalCode'))),
            _part('p4', 'join on my address lines', 50, 'single_column', column='AddressLine1'),
            _part('p5', 'have a column correlating with my StateProvinceID joined on AddressLine1', 50, 'correlation',
                  join='AddressLine1', target='StateProvinceID'))), k=5, status='certain')),
    'dependent_data': Example(
        EMPLOYEE, 'Tables that join on my StateProvinceName column and separately on my CountryRegionName column, or '
                  'on my EmailAddress column and separately on my PhoneNumber column',
        _plan(Combiner(id='c1', kind='union', k=200, inputs=(
            Combiner(id='c2', kind='intersection', k=100, inputs=(
                _part('p1', 'StateProvinceName', 100, 'single_column', column='StateProvinceName'),
                _part('p2', 'CountryRegionName', 100, 'single_column', column='CountryRegionName'))),
            Combiner(id='c3', kind='intersection', k=100, inputs=(
                _part('p3', 'EmailAddress', 100, 'single_column', column='EmailAddress'),
                _part('p4', 'PhoneNumber', 100, 'single_column', column='PhoneNumber')))))),
        template='dependent_data', slots={'first': ('StateProvinceName', 'CountryRegionName'),
                                          'second': ('EmailAddress', 'PhoneNumber')}),
    'negative_examples': Example(
        ADDRESS, 'Tables with my cities together with their postal codes, but not tables that have the city Bothell '
                 'with state province 79 or Seattle with 79',
        _plan(Combiner(id='c1', kind='difference', removal='exclusion', inputs=(
            _part('p1', 'Tables with my cities together with their postal codes', 100, 'multi_column',
                  columns=('City', 'PostalCode')),
            _part('p2', 'tables that have the city Bothell with state province 79 or Seattle with 79', 100,
                  'multi_column', rows=NEGATIVE_ROWS, columns=('City', 'StateProvinceID'))))),
        template='negative_examples',
        slots={'exclusive': ('City', 'PostalCode'), 'inclusive': ('City', 'StateProvinceID'),
               'inclusive_rows': NEGATIVE_ROWS}),
    'feature_for_ml': Example(
        ADDRESS, 'Leave out the tables whose numbers correlate best with my AddressID joined on City, and find '
                 'tables with a column that correlates with my StateProvinceID joined on City',
        _plan(Combiner(id='c1', kind='difference', removal='comparison', k=100, inputs=(
            _part('p1', 'tables with a column that correlates with my StateProvinceID joined on City', 100,
                  'correlation', join='City', target='StateProvinceID'),
            _part('p2', 'the tables whose numbers correlate best with my AddressID joined on City', 10,
                  'correlation', join='City', target='AddressID'))))),
    'collinearity': Example(
        EMPLOYEE, 'Tables that contain my titles with first names and have a column correlating with my '
                  'BusinessEntityID joined on MiddleName, except the best correlations with my EmailPromotion joined '
                  'on MiddleName',
        _plan(Combiner(id='c1', kind='intersection', inputs=(
            _part('p1', 'contain my titles with first names', 100, 'multi_column', columns=('Title', 'FirstName')),
            Combiner(id='c2', kind='difference', removal='comparison', k=100, inputs=(
                _part('p2', 'have a column correlating with my BusinessEntityID joined on MiddleName', 100,
                      'correlation', join='MiddleName', target='BusinessEntityID'),
                _part('p3', 'the best correlations with my EmailPromotion joined on MiddleName', 10,
                      'correlation', join='MiddleName', target='EmailPromotion'))))))),
    'columns_together': Example(
        CONTACTS, 'Tables that match my contacts on first name and last name together, in one row',
        _plan(_part('p1', 'Tables that match my contacts on first name and last name together, in one row', None,
                    'multi_column', columns=('FirstName', 'LastName'))),
        template='multi_column_join', slots={'columns': ('FirstName', 'LastName')}),
    'plain_join': Example(EMPLOYEE, 'Tables that join with my employees on City and StateProvinceName', _plan(
        _part('p1', 'Tables that join with my employees on City and StateProvinceName', None, 'multi_column',
              columns=('City', 'StateProvinceName'))), template='multi_column_join',
        slots={'columns': ('City', 'StateProvinceName')}),
    'join_in_correlation': Example(
        TERRITORIES, 'Tables I can join with on my territory names that have a numeric column correlating with my '
                     'SalesYTD',
        _plan(_part('p1', 'Tables I can join with on my territory names that have a numeric column correlating with my '
                          'SalesYTD', 100, 'correlation', join='Name', target='SalesYTD')),
        template='correlation', slots={'join': 'Name', 'target': 'SalesYTD'}),
    'ambiguous_column': Example(CONTACTS, 'Tables that join on the names in my table', _plan(
        _part('p1', 'Tables that join on the names in my table', 100, 'single_column', column=NAMES)),
        template='single_column_join', slots={'column': NAMES}),
    'unnamed_column': Example(CONTACTS, 'Tables I can join my contact list with', _plan(
        _part('p1', 'Tables I can join my contact list with', 100, 'single_column', column=UNNAMED_COLUMN)),
        template='single_column_join', slots={'column': UNNAMED_COLUMN}),
    'about_paris': Example(
        ADDRESS, 'Tables about Paris',
        _plan(_part('p1', 'Tables about Paris', None, 'keyword', words=('Paris',)).model_copy(
            update={'seeker_status': 'ambiguous'})),
        alternatives=(PARIS_JOIN,), readings=(PARIS_JOIN,), template='keyword', slots={'words': ('Paris',)}),
    'keyword_exclusion': Example(
        ADDRESS, 'Leaving out tables that mention Paris, which tables join on my cities?',
        _plan(Combiner(id='c1', kind='difference', removal='exclusion', k=100, inputs=(
            _part('p1', 'which tables join on my cities', 100, 'single_column', column='City'),
            _part('p2', 'tables that mention Paris', 1000, 'keyword', words=('Paris',)))))),
    'keyword_words_together': Example(
        TERRITORIES, 'Tables that join on my country region codes, except the ones that contain Germany or Australia',
        _plan(Combiner(id='c1', kind='difference', removal='exclusion', k=100, inputs=(
            _part('p1', 'Tables that join on my country region codes', 100, 'single_column',
                  column='CountryRegionCode'),
            _part('p2', 'the ones that contain Germany or Australia', 1000, 'keyword',
                  words=('Germany', 'Australia')))))),
    'open_direction': Example(
        TERRITORIES, 'Of the tables with a column correlating with my SalesLastYear joined on CountryRegionCode and the '
                     'tables that mention Europe, take the shared ones out of the list',
        _plan(Combiner(id='c1', kind='difference', removal='exclusion', k=100,
                       inputs=(DIRECTION_KEPT, DIRECTION_REMOVED))),
        readings=(Combiner(id='c1', kind='difference', removal='exclusion',
                           inputs=(DIRECTION_REMOVED, DIRECTION_KEPT)),)),
    'exclusion_or_comparison': Example(
        EMPLOYEE, 'Tables that join on my City column, apart from the tables a search for Redmond turns up',
        _plan(Combiner(id='c1', kind='difference', removal='exclusion', k=100, inputs=(REMOVAL_KEPT, REMOVAL_REMOVED))),
        readings=(Combiner(id='c1', kind='difference', removal='comparison',
                           inputs=(REMOVAL_KEPT, REMOVAL_REMOVED)),)),
    'top_k': Example(EMPLOYEE, 'Top 20 tables I can join with on my job titles', _plan(
        _part('p1', 'tables I can join with on my job titles', 200, 'single_column', column='JobTitle'), k=20,
        status='certain'), template='single_column_join', slots={'column': 'JobTitle'}),
    'imputation': Example(
        DEMOGRAPHICS, 'Occupation is blank for some of the people in my table. Find tables that could supply it.',
        _plan(Combiner(id='c1', kind='intersection', inputs=(
            _part('p1', 'Find tables that could supply it', 100, 'multi_column',
                  columns=('BusinessEntityID', 'Occupation'), rows_where={'column': 'Occupation', 'cell': 'value'}),
            _part('p2', 'Find tables that could supply it', 100, 'single_column', column=UNNAMED_KEY,
                  rows_where={'column': 'Occupation', 'cell': 'empty'})))),
        template='imputation', slots={'key': UNNAMED_KEY, 'fill': 'Occupation'}),
    'more_rows': Example(
        CONTACTS, 'I want more contacts like these. Which tables could extend my list with further rows?',
        _plan(Combiner(id='c1', kind='union', k=300, inputs=(
            _part('p1', 'extend my list with further rows', 100, 'single_column', column=_unnamed('FirstName')),
            _part('p2', 'extend my list with further rows', 100, 'single_column', column=_unnamed('LastName')),
            _part('p3', 'extend my list with further rows', 100, 'single_column', column=_unnamed('EmailAddress'))))),
        template='union_search', slots={'columns': _unnamed(('FirstName', 'LastName', 'EmailAddress'))}),
    'features_to_predict': Example(
        TERRITORIES, 'Which features would help predict my SalesYTD beyond what SalesLastYear already tells me, '
                     'matched on the territory name?',
        _plan(Combiner(id='c1', kind='difference', removal='comparison', k=100, inputs=(
            _part('p1', 'help predict my SalesYTD', 100, 'correlation', join='Name', target='SalesYTD'),
            _part('p2', 'beyond what SalesLastYear already tells me', 10, 'correlation', join='Name',
                  target='SalesLastYear')))),
        template='feature_for_ml', slots={'join': 'Name', 'target': 'SalesYTD', 'feature': 'SalesLastYear'}),
    'separate_joins': Example(
        CONTACTS, 'Tables that join on my phone numbers and, as a separate match, on my email addresses',
        _plan(Combiner(id='c1', kind='intersection', k=100, inputs=(
            _part('p1', 'join on my phone numbers', 100, 'single_column', column='PhoneNumber'),
            _part('p2', 'on my email addresses', 100, 'single_column', column='EmailAddress'))))),
    'join_with_constraint': Example(
        CONTACTS, 'Tables that join on my store names, but only the ones that also mention Purchasing',
        _plan(Combiner(id='c1', kind='intersection', k=100, inputs=(
            _part('p1', 'Tables that join on my store names', 100, 'single_column', column='Name'),
            _part('p2', 'the ones that also mention Purchasing', 100, 'keyword', words=('Purchasing',)))))),
    'task_without_word': Example(
        CANDIDATES, 'Most of my job candidates have no BusinessEntityID yet. Find tables that could supply it, except '
                    'tables that mention Purchasing.',
        _in_tree(Combiner(id='c1', kind='difference', removal='exclusion', inputs=(
            Combiner(id='c2', kind='intersection', k=100, inputs=(
                _part('p2', TASK_WORDS, 100, 'multi_column', columns=('JobCandidateID', 'BusinessEntityID'),
                      rows_where={'column': 'BusinessEntityID', 'cell': 'value'}),
                _part('p3', TASK_WORDS, 100, 'single_column', column=_unnamed('JobCandidateID'),
                      rows_where={'column': 'BusinessEntityID', 'cell': 'empty'}))),
            _part('p1', 'tables that mention Purchasing', 1000, 'keyword', words=('Purchasing',)))),
            'imputation', ('p2', 'p3')),
        template='imputation', slots={'key': _unnamed('JobCandidateID'), 'fill': 'BusinessEntityID'}),
    'task_with_join': Example(
        DESCRIPTIONS, 'I want more rows like my product descriptions to add to this table, from tables that also join '
                      'on my CultureID values.',
        _in_tree(Combiner(id='c1', kind='intersection', k=300, inputs=(
            Combiner(id='c2', kind='union', k=300, inputs=(
                _part('p2', MORE_ROWS, 100, 'single_column', column=_unnamed('Name')),
                _part('p3', MORE_ROWS, 100, 'single_column', column=_unnamed('ProductModel')),
                _part('p4', MORE_ROWS, 100, 'single_column', column=_unnamed('Description')))),
            _part('p1', 'tables that also join on my CultureID values', 100, 'single_column', column='CultureID'))),
            'union_search', ('p2', 'p3', 'p4')),
        template='union_search', slots={'columns': _unnamed(('Name', 'ProductModel', 'Description'))}),
}

DECOMPOSER = ('single_column_join', 'pairs_and_join', 'union_search', 'complex_search', 'dependent_data',
              'negative_examples', 'keyword_exclusion', 'feature_for_ml', 'collinearity', 'columns_together',
              'plain_join', 'join_in_correlation', 'keyword_words_together', 'top_k', 'open_direction',
              'exclusion_or_comparison', 'typed_rows', 'about_paris', 'imputation', 'more_rows', 'features_to_predict',
              'separate_joins', 'join_with_constraint', 'task_without_word', 'task_with_join')

SEEKERS = (('single_column_join', 'p1'), ('multi_column_join', 'p1'), ('keyword_search', 'p1'),
           ('correlation_search', 'p1'), ('nl_search', 'p1'))
STATUSES = (('ambiguous_column', 'p1'), ('unnamed_column', 'p1'), ('about_paris', 'p1'))


def filler_parts(role: Planner.Role) -> List[Tuple[Example, Part]]:
    parts = {(name, part.id): (example, part) for name, example in EXAMPLES.items() for part in example.plan.parts()}
    base, by_seeker = SEEKERS + STATUSES, {}
    for name, example in EXAMPLES.items():
        for part_id, part_role in Planner.roles(example.plan.root).items():
            if (name, part_id) not in base and part_role == role and part_id not in example.plan.template_parts:
                by_seeker.setdefault(parts[name, part_id][1].fill.seeker, parts[name, part_id])
    return [parts[key] for key in base] + list(by_seeker.values())[:2]

import ipywidgets as widgets
import datetime as dt

class DateRangePicker:
    _from_date_value = None
    _to_date_value = None

    def __init__(self, handler, default_from_date=dt.datetime.now().date(), default_to_date=dt.datetime.now().date() + dt.timedelta(days=1)) -> None:
        self._handler = handler

        self.from_date = widgets.DatePicker(
            description='From',
            disabled=False,
            value=default_from_date
        )
        self.to_date = widgets.DatePicker(
            description='To',
            disabled=False,
            value=default_to_date
        )

        self.from_date.observe(self.observe, names="value")
        self.to_date.observe(self.observe, names="value")

    def observe(self, obj):
        if isinstance(obj['owner'], str) and obj['owner'] == self.from_date.description:
            self._from_date_value = obj['new']
        elif isinstance(obj['owner'], widgets.DatePicker) and obj['owner'].description == self.from_date.description:
            self._from_date_value = obj['new']
        elif isinstance(obj['owner'], str) and obj['owner'] == self.to_date.description:
            self._to_date_value = obj['new']
        elif isinstance(obj['owner'], widgets.DatePicker) and obj['owner'].description == self.to_date.description:
            self._to_date_value = obj['new']

        if all([self._from_date_value, self._to_date_value]):
            self._handler(self._from_date_value, self._to_date_value)
    
    def get_widget(self):
        return self.from_date, self.to_date

    def trigger(self):
        def get_event(owner, date):
            return {
                'name': 'value', 
                'old': None, 
                'new': date, 
                'type': 'change',
                'owner': owner
            }
        self.from_date.notify_change(get_event(self.from_date.description, self.from_date.value))
        self.to_date.notify_change(get_event(self.to_date.description, self.to_date.value))

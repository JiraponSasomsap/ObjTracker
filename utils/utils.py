def conv2dataDict(**data):
        '''
        Convert column-based keyword arguments (dict of lists) to a list of row-based dicts.

        Example:
            conv2dataDict(x=[1,2], y=[3,4]) 
            => [{'x': 1, 'y': 3}, {'x': 2, 'y': 4}]

        Parameters:
            **data: Keyword arguments where each value is a list of the same length.

        Returns:
            List[dict]: List of dictionaries representing each row.

        Raises:
            ValueError: If input lists are not all the same length.
        '''
        keys = list(data.keys())
        vals = list(data.values())

        for i in range(1, len(vals)):
            if len(vals[0]) != len(vals[i]):
                raise ValueError("All values must have the same length")
            
        data_list = []

        for i in range(len(vals[0])):
            row = {k : data[k][i] for k in keys}
            data_list.append(row)
        
        return data_list
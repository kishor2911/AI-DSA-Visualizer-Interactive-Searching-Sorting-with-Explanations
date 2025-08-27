import streamlit as st
import numpy as np
import time

# --- Helper functions for algorithms ---
def swap(arr, i, j):
    """Swaps two elements in an array."""
    arr[i], arr[j] = arr[j], arr[i]

# --- Sorting Algorithms Visualizers ---

def bubble_sort_visualizer(arr):
    """Performs Bubble Sort and yields intermediate steps."""
    n = len(arr)
    # Yielding -1 for 'i' to indicate no outer loop pass yet
    yield list(arr), -1, -1, -1, f"**Initialize:** Starting Bubble Sort on array with {n} elements.", False

    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            yield list(arr), i, j, j+1, f"**Pass {i+1}:** Comparing `{arr[j]}` and `{arr[j+1]}`.", False
            if arr[j] > arr[j + 1]:
                swap(arr, j, j + 1)
                swapped = True
                yield list(arr), i, j, j+1, f"**Pass {i+1}:** Swapping `{arr[j]}` and `{arr[j+1]}`.", True
        
        yield list(arr), i, n - 1 - i, n - 1 - i, f"**Pass {i+1}:** Element `{arr[n - 1 - i]}` is now in its final position.", False
        
        if not swapped:
            yield list(arr), -1, -1, -1, "**Optimization:** No swaps were made. Array is already sorted.", False
            break
    
    yield list(arr), -1, -1, -1, "**Sort Complete:** The array is fully sorted.", False

def selection_sort_visualizer(arr):
    """Performs Selection Sort and yields intermediate steps."""
    n = len(arr)
    yield list(arr), -1, -1, -1, f"**Initialize:** Starting Selection Sort on array with {n} elements.", False

    for i in range(n):
        min_idx = i
        yield list(arr), i, i, min_idx, f"**Pass {i+1}:** Assuming element at index `{i}` (`{arr[i]}`) is the minimum.", False

        for j in range(i + 1, n):
            yield list(arr), i, i, j, f"**Pass {i+1}:** Comparing current minimum (`{arr[min_idx]}`) with `{arr[j]}`.", False
            if arr[j] < arr[min_idx]:
                min_idx = j
                yield list(arr), i, i, min_idx, f"**Pass {i+1}:** New minimum found: `{arr[min_idx]}` at index `{min_idx}`.", False
        
        if min_idx != i:
            swap(arr, i, min_idx)
            yield list(arr), i, i, min_idx, f"**Pass {i+1}:** Swapping minimum element `{arr[min_idx]}` with element at index `{i}`.", True
        else:
            yield list(arr), i, i, min_idx, f"**Pass {i+1}:** No swap needed, element is already in place.", False

    yield list(arr), -1, -1, -1, "**Sort Complete:** The array is fully sorted.", False

def insertion_sort_visualizer(arr):
    """Performs Insertion Sort and yields intermediate steps."""
    n = len(arr)
    yield list(arr), -1, -1, -1, "**Initialize:** Starting Insertion Sort. The first element is considered sorted.", False

    for i in range(1, n):
        key = arr[i]
        j = i - 1
        yield list(arr), i, i, j, f"**Pass {i}:** Selecting key `{key}` at index `{i}`.", False
        
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            yield list(arr), i, j+1, i, f"**Pass {i}:** Shifting element `{arr[j + 1]}` right to make space for `{key}`.", False

        arr[j + 1] = key
        yield list(arr), i, j + 1, i, f"**Pass {i}:** Inserting key `{key}` into its correct position at index `{j + 1}`.", True

    yield list(arr), -1, -1, -1, "**Sort Complete:** The array is fully sorted.", False

def merge_sort_visualizer(arr, l, r):
    """Performs Merge Sort and yields intermediate steps."""
    if l < r:
        m = (l + r) // 2
        yield list(arr), l, r, m, f"**Divide:** Splitting array from `[{l}, {r}]` into two halves: `[{l}, {m}]` and `[{m+1}, {r}]`.", False
        yield from merge_sort_visualizer(arr, l, m)
        yield from merge_sort_visualizer(arr, m + 1, r)
        
        L = arr[l:m + 1]
        R = arr[m + 1:r + 1]
        i = j = 0
        k = l
        
        yield list(arr), l, r, m, f"**Merge:** Merging two subarrays.", False
        
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            yield list(arr), l, r, m, "Merging elements...", False

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            yield list(arr), l, r, m, "Copying remaining elements from left subarray.", False

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            yield list(arr), l, r, m, "Copying remaining elements from right subarray.", False
    
    if l == 0 and r == len(arr) - 1:
        yield list(arr), -1, -1, -1, "**Sort Complete:** The array is fully sorted.", False

def partition_visualizer(arr, low, high):
    """A helper function for Quick Sort visualization."""
    i = (low - 1)
    pivot = arr[high]
    yield list(arr), low, high, high, f"**Partition:** Selecting pivot `{pivot}` at index `{high}`.", False

    for j in range(low, high):
        yield list(arr), low, high, j, f"**Partition:** Comparing element at index `{j}` (`{arr[j]}`) with pivot `{pivot}`.", False
        if arr[j] <= pivot:
            i += 1
            swap(arr, i, j)
            yield list(arr), low, high, j, f"**Partition:** Swapping `{arr[i]}` with `{arr[j]}`.", True

    swap(arr, i + 1, high)
    yield list(arr), low, high, i + 1, f"**Partition:** Placing pivot `{pivot}` at its correct position `{i + 1}`.", True
    return i + 1

def quick_sort_visualizer(arr, low, high):
    """Performs Quick Sort and yields intermediate steps."""
    if low < high:
        pi = yield from partition_visualizer(arr, low, high)
        yield from quick_sort_visualizer(arr, low, pi - 1)
        yield from quick_sort_visualizer(arr, pi + 1, high)
    
    if low == 0 and high == len(arr) - 1:
        yield list(arr), -1, -1, -1, "**Sort Complete:** The array is fully sorted.", False

def heapify(arr, n, i, visualizer):
    """A helper function to heapify a subtree in Heap Sort."""
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        swap(arr, i, largest)
        yield list(arr), largest, i, f"Heapify: Swapping `{arr[largest]}` and `{arr[i]}` to maintain heap property.", True
        yield from visualizer(arr, n, largest, visualizer)

def heap_sort_visualizer(arr):
    """Performs Heap Sort and yields intermediate steps."""
    n = len(arr)
    
    yield list(arr), -1, -1, f"**Phase 1: Build Max-Heap.** Building a max-heap from the input array.", False
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(arr, n, i, heapify)

    yield list(arr), -1, -1, f"**Phase 2: Extract Elements.** Extracting elements one by one from the heap.", False
    for i in range(n - 1, 0, -1):
        swap(arr, i, 0)
        yield list(arr), i, 0, f"Swapping root `{arr[i]}` with last element `{arr[0]}`.", True
        yield from heapify(arr, i, 0, heapify)

    yield list(arr), -1, -1, "**Sort Complete:** The array is fully sorted.", False

def counting_sort_visualizer(arr):
    """Performs Counting Sort and yields intermediate steps."""
    output = [0] * len(arr)
    max_val = max(arr) if arr else 0
    count = [0] * (max_val + 1)

    yield list(arr), -1, -1, f"**Initialize:** Creating a count array of size `{max_val + 1}`.", False

    for i in range(len(arr)):
        count[arr[i]] += 1
        yield list(arr), i, -1, f"**Step 1: Counting.** Incrementing count for element `{arr[i]}`. Count array: `{count}`.", False

    for i in range(1, len(count)):
        count[i] += count[i - 1]
        yield list(arr), -1, -1, f"**Step 2: Cumulative Sum.** Updating cumulative count at index `{i}`. Count array: `{count}`.", False

    i = len(arr) - 1
    while i >= 0:
        val = arr[i]
        pos = count[val] - 1
        output[pos] = val
        count[val] -= 1
        yield list(arr), i, pos, f"**Step 3: Placing.** Placing `{val}` from original array into `output` array at index `{pos}`.", False
        i -= 1

    for i in range(len(arr)):
        arr[i] = output[i]
        
    yield list(arr), -1, -1, "**Sort Complete:** Final sorted array copied from output array.", False

def counting_sort_radix(arr, exp):
    """A helper function for Radix Sort visualization."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = (arr[i] // exp)
        count[int(index % 10)] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp)
        output[count[int(index % 10)] - 1] = arr[i]
        count[int(index % 10)] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]
    
    yield list(arr), -1, -1, f"**Phase:** Sorted by digit `{exp}`.", False

def radix_sort_visualizer(arr):
    """Performs Radix Sort and yields intermediate steps."""
    max_val = max(arr) if arr else 0
    exp = 1

    yield list(arr), -1, -1, f"**Initialize:** Starting Radix Sort. Max value is `{max_val}`.", False

    while max_val // exp > 0:
        yield from counting_sort_radix(arr, exp)
        exp *= 10
    
    yield list(arr), -1, -1, "**Sort Complete:** The array is fully sorted.", False

def bucket_sort_visualizer(arr):
    """Performs Bucket Sort and yields intermediate steps."""
    n = len(arr)
    buckets = [[] for _ in range(n)]
    max_val = max(arr) if arr else 1
    
    yield list(arr), -1, -1, f"**Initialize:** Starting Bucket Sort with `{n}` buckets. Max value is `{max_val}`.", False

    for i in arr:
        index = min(n - 1, int((i * n) / (max_val + 1)))
        buckets[index].append(i)
        yield list(arr), arr.index(i), -1, f"**Distribution:** Placing `{i}` into bucket `{index}`.", False
        
    for i in range(n):
        yield list(arr), -1, -1, f"**Sorting:** Sorting bucket `{i}`.", False
        buckets[i].sort()
    
    k = 0
    for i in range(n):
        for j in range(len(buckets[i])):
            arr[k] = buckets[i][j]
            k += 1
            yield list(arr), k-1, i, f"**Gathering:** Copying sorted element `{arr[k-1]}` from bucket `{i}`.", False
            
    yield list(arr), -1, -1, "**Sort Complete:** The array is fully sorted.", False

def shell_sort_visualizer(arr):
    """Performs Shell Sort and yields intermediate steps."""
    n = len(arr)
    gap = n // 2
    
    yield list(arr), -1, -1, f"**Initialize:** Starting Shell Sort. Initial gap is `{gap}`.", False

    while gap > 0:
        yield list(arr), -1, -1, f"**Phase: Gap = {gap}.** Performing gapped insertion sort.", False
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                yield list(arr), -1, j, i, f"Shifting element at index `{j-gap}` to `{j}`.", False
            
            arr[j] = temp
            yield list(arr), -1, j, i, f"Inserting `{temp}` into its correct position with gap `{gap}`.", True
        
        gap //= 2
    
    yield list(arr), -1, -1, "**Sort Complete:** The array is fully sorted.", False

# --- Search Algorithm Visualizers ---

def linear_search_visualizer(arr, target):
    """Performs linear search on an array and yields intermediate steps."""
    if not arr:
        yield -1, False, "**Search Complete:** Array is empty. Target not found."
        return

    yield -1, False, f"**Initialize:** Starting linear search. Array: `{arr}`. Target: **{target}**."
    
    for i, val in enumerate(arr):
        yield i, False, f"**Comparing:** Element at index **{i}** (value: **{val}**) with target **{target}**."
        if val == target:
            yield i, True, f"**Match Found!** Target **{target}** found at index **{i}**."
            return i
    
    yield -1, False, f"**Search Complete:** All elements checked. Target **{target}** not found."
    return -1

def binary_search_visualizer(arr, target):
    """Performs binary search on a sorted array and yields intermediate steps."""
    if not arr:
        yield -1, -1, -1, False, "**Search Complete:** Array is empty. Target not found."
        return

    low, high = 0, len(arr) - 1
    yield low, high, -1, False, f"**Initialize:** Starting binary search. Array: `{arr}`. Target: **{target}**."
    
    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid]
        yield low, high, mid, False, f"**Current Range:** `low={low}`, `high={high}`. **Mid:** `arr[{mid}] = {mid_val}`."
        
        if mid_val == target:
            yield low, high, mid, True, f"**Match Found!** Target **{target}** found at index **{mid}**!"
            return mid
        elif mid_val < target:
            low = mid + 1
            yield low, high, mid, False, f"**Comparison:** Mid is less than target. Discarding left half. New range: `[{low}, {high}]`."
        else:
            high = mid - 1
            yield low, high, mid, False, f"**Comparison:** Mid is greater than target. Discarding right half. New range: `[{low}, {high}]`."
            
    yield -1, -1, -1, False, f"**Search Complete:** `low > high` (range empty). Target not found."
    return -1

def jump_search_visualizer(arr, target):
    """Performs jump search on a sorted array and yields intermediate steps."""
    n = len(arr)
    if not arr:
        yield -1, -1, -1, False, "**Search Complete:** Array is empty. Target not found."
        return

    block_size = int(np.sqrt(n))
    prev = 0
    
    yield prev, -1, -1, False, f"**Initialize:** Starting jump search with block size `{block_size}`."
    
    while prev < n and arr[min(block_size, n) - 1] < target:
        jump_point = min(block_size, n) - 1
        yield prev, jump_point, -1, False, f"**Jumping:** Value at `{jump_point}` (`{arr[jump_point]}` is less than target `{target}`. Moving to next block."
        prev = block_size
        block_size += int(np.sqrt(n))
    
    if prev >= n:
        yield -1, -1, -1, False, "**Search Complete:** Jumped past the end. Target not found."
        return -1
    
    curr = prev
    while curr < min(block_size, n):
        if curr == prev:
            yield prev, min(block_size, n) - 1, -1, False, f"**Linear Scan:** Block found. Starting linear search from `{prev}` to `{min(block_size, n)-1}`."
        
        yield prev, min(block_size, n) - 1, curr, False, f"**Linear Scan:** Checking index `{curr}` (value `{arr[curr]}`)."
        if arr[curr] == target:
            yield prev, min(block_size, n) - 1, curr, True, f"**Match Found!** Target found at index `{curr}`."
            return curr
        curr += 1
    
    yield -1, -1, -1, False, "**Search Complete:** All elements checked. Target not found."
    return -1

def interpolation_search_visualizer(arr, target):
    """Performs interpolation search on a sorted, uniformly distributed array."""
    if not arr:
        yield -1, -1, -1, False, "**Search Complete:** Array is empty. Target not found."
        return
        
    low, high = 0, len(arr) - 1
    yield low, high, -1, False, f"**Initialize:** Starting interpolation search. Range `[{low}, {high}]`."
    
    while low <= high and target >= arr[low] and target <= arr[high]:
        if arr[high] == arr[low]:
            pos = low if target == arr[low] else -1
        else:
            pos = low + int(((float(high - low) / (arr[high] - arr[low])) * (target - arr[low])))
        
        pos = max(low, min(high, pos))
        
        if pos < 0 or pos >= len(arr):
             yield -1, -1, -1, False, "**Search Complete:** Probe position out of bounds. Target not found."
             return -1

        yield low, high, pos, False, f"**Probe:** Calculating probe position `{pos}`. Value at probe is `{arr[pos]}`."
        
        if arr[pos] == target:
            yield low, high, pos, True, f"**Match Found!** Target found at index `{pos}`."
            return pos
        elif arr[pos] < target:
            low = pos + 1
            yield low, high, pos, False, "Probe value is less than target. Shifting search to the right."
        else:
            high = pos - 1
            yield low, high, pos, False, "Probe value is greater than target. Shifting search to the left."
            
    yield -1, -1, -1, False, "**Search Complete:** Target not found."
    return -1

def exponential_search_visualizer(arr, target):
    """Performs exponential search on a sorted array."""
    if not arr:
        yield -1, -1, -1, -1, False, "**Search Complete:** Array is empty. Target not found."
        return

    if arr[0] == target:
        yield 0, 0, 0, 0, True, "**Match Found!** Target found at index 0."
        return 0

    bound = 1
    yield bound, -1, -1, -1, False, "**Phase 1: Finding Range.** Starting with bound 1."
    
    while bound < len(arr) and arr[bound] < target:
        prev_bound = bound
        bound *= 2
        yield bound, -1, -1, -1, False, f"Bound {prev_bound} is too small. Doubling to `{bound}`."
    
    low = bound // 2
    high = min(bound, len(arr) - 1)
    
    yield bound, low, high, -1, False, f"**Phase 2: Binary Search.** Range found: `[{low}, {high}]`. Starting binary search."

    for l, h, m, f, msg in binary_search_visualizer(arr[low:high+1], target):
        absolute_m = m + low if m != -1 else -1
        yield bound, l + low, h + low, absolute_m, f, msg
        if f:
            return absolute_m

    yield -1, -1, -1, -1, False, "**Search Complete:** Target not found."
    return -1

def ternary_search_visualizer(arr, target):
    """Performs ternary search on a sorted array and yields intermediate steps."""
    if not arr:
        yield -1, -1, -1, -1, False, "**Search Complete:** Array is empty. Target not found."
        return
        
    low, high = 0, len(arr) - 1
    yield low, high, -1, -1, False, f"**Initialize:** Starting ternary search. Range `[{low}, {high}]`."
    
    while low <= high:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3

        if mid1 == mid2 and low != high:
            if arr[low] == target:
                yield low, high, low, high, True, "**Match Found!**"
                return low
            if arr[high] == target:
                yield low, high, low, high, True, "**Match Found!**"
                return high
            yield low, high, low, high, False, "**Search Complete:** Target not found."
            return -1

        yield low, high, mid1, mid2, False, f"**Division:** Dividing array into three parts with midpoints at `{mid1}` and `{mid2}`."

        if arr[mid1] == target:
            yield low, high, mid1, mid2, True, f"**Match Found!** Target found at `{mid1}`."
            return mid1
        if arr[mid2] == target:
            yield low, high, mid1, mid2, True, f"**Match Found!** Target found at `{mid2}`."
            return mid2
        
        if target < arr[mid1]:
            high = mid1 - 1
            yield low, high, mid1, mid2, False, "Target is in the first third. Updating range."
        elif target > arr[mid2]:
            low = mid2 + 1
            yield low, high, mid1, mid2, False, "Target is in the third third. Updating range."
        else:
            low = mid1 + 1
            high = mid2 - 1
            yield low, high, mid1, mid2, False, "Target is in the middle third. Updating range."

    yield -1, -1, -1, -1, False, "**Search Complete:** Target not found."
    return -1

# --- Streamlit App Layout and Logic ---

st.set_page_config(layout="wide", page_title="Algorithms Visualizer")
st.markdown("""
<style>
    /* Main body and text colors for a modern dark theme */
    .main {
        background-color: #0d1117; /* GitHub dark theme background */
        color: #c9d1d9; /* Lighter text color for contrast */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }

    /* Headings with a touch of elegance */
    h1, h2, h3 {
        color: #58a6ff; /* A vibrant blue accent */
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }

    /* Input elements styling */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #c9d1d9;
    }

    /* Button styling with a clean gradient and hover effect */
    .stButton > button {
        background-image: linear-gradient(to right, #4c51bf, #6b46c1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Array container with a subtle, elegant border */
    .array-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px; /* Increased gap for better spacing */
        padding: 15px;
        border: 1px solid #2f3336;
        border-radius: 12px;
        background-color: #161b22; /* A slightly darker background for the container */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    /* Individual array element styling */
    .array-element {
        padding: 8px 12px;
        border-radius: 6px;
        color: #e0e0e0;
        min-width: 40px; /* Slightly wider for better visual balance */
        text-align: center;
        font-weight: 500;
        font-size: 1.1rem;
        background-color: #21262d; /* Default element color */
        transition: all 0.3s ease;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05);
    }

    /* Algorithm visualization colors - refined palette */
    .compare-element {
        background-color: #d1b827; /* A rich yellow for comparison */
        color: #21262d;
        font-weight: bold;
        transform: scale(1.1);
        box-shadow: 0 0 10px #d1b827;
    }

    .swap-element {
        background-color: #ff7b72; /* A vibrant red for swaps */
        font-weight: bold;
        transform: scale(1.2);
        box-shadow: 0 0 15px #ff7b72;
    }

    .sorted-element {
        background-color: #3fb950; /* A pleasant green for sorted elements */
        font-weight: bold;
        transform: scale(1.2);
        box-shadow: 0 0 15px #3fb950;
    }

    .active-element {
        background-color: #3081e8; /* A bright blue for the active range/element */
        color: white;
        font-weight: bold;
        box-shadow: 0 0 10px #3081e8;
    }
    
    .found-element {
        background-color: #20bf6b; /* Same as sorted, as finding is the end state */
        font-weight: bold;
        transform: scale(1.2);
        box-shadow: 0 0 15px #20bf6b;
    }

    /* Info box styling for messages and logs */
    .info-box {
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 20px;
        background-color: #161b22;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("Sorting & Searching Algorithms Visualizer 📈")
st.markdown("### Choose an algorithm type and method from the sidebar to visualize.")

# --- Sidebar for user inputs and controls ---
st.sidebar.header("Configuration")
selected_category = st.sidebar.selectbox("Select Algorithm Type", ("Sorting", "Searching"))

if selected_category == "Sorting":
    algorithm_choice = st.sidebar.selectbox(
        "Select Sorting Algorithm",
        ("Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", "Quick Sort",
         "Heap Sort", "Counting Sort", "Radix Sort", "Bucket Sort", "Shell Sort")
    )
    default_data = "64, 34, 25, 12, 22, 11, 90"
    data_input = st.sidebar.text_input("Enter numbers (comma-separated)", default_data)
    
    col_start_sort, col_stop_sort = st.sidebar.columns(2)
    with col_start_sort:
        if st.button("Visualize Sort"):
            try:
                data_list = [int(x.strip()) for x in data_input.split(',') if x.strip()]
                if not data_list:
                    st.warning("Please enter some numbers to visualize.")
                else:
                    st.session_state.sort_data = data_list[:]
                    st.session_state.sort_algo = algorithm_choice
                    st.session_state.run_sort = True
                    st.session_state.continue_sort = True
                    st.session_state.current_message_log = []
                    if 'run_search' in st.session_state:
                        st.session_state.run_search = False
            except ValueError:
                st.error("Invalid input. Please enter comma-separated integers.")

    with col_stop_sort:
        if st.button("Stop Sort"):
            st.session_state.continue_sort = False
            st.warning("Visualization paused. Press 'Continue' to resume.")

    if 'continue_sort' in st.session_state and not st.session_state.continue_sort:
        if st.sidebar.button("Continue Sort"):
            st.session_state.continue_sort = True

elif selected_category == "Searching":
    algorithm_choice = st.sidebar.selectbox(
        "Select Search Algorithm",
        ("Linear Search", "Binary Search", "Jump Search", "Interpolation Search", "Exponential Search", "Ternary Search")
    )
    default_data = "1, 5, 7, 10, 12, 15, 20, 22, 25, 30, 35, 40, 45, 50"
    data_input = st.sidebar.text_input("Enter numbers (comma-separated)", default_data)
    target_input = st.sidebar.number_input("Enter target value", value=30, step=1)
    
    col_start_search, col_stop_search = st.sidebar.columns(2)
    with col_start_search:
        if st.button("Visualize Search"):
            try:
                data_list = [int(x.strip()) for x in data_input.split(',') if x.strip()]
                if not data_list:
                    st.warning("Please provide valid numbers to visualize.")
                else:
                    st.session_state.search_data = data_list[:]
                    st.session_state.search_target = target_input
                    st.session_state.search_algo = algorithm_choice
                    st.session_state.run_search = True
                    st.session_state.continue_search = True
                    st.session_state.current_message_log = []
                    if 'run_sort' in st.session_state:
                        st.session_state.run_sort = False
            except ValueError:
                st.error("Invalid input. Please enter comma-separated integers.")

    with col_stop_search:
        if st.button("Stop Search"):
            st.session_state.continue_search = False

    if 'continue_search' in st.session_state and not st.session_state.continue_search:
        if st.sidebar.button("Continue Search"):
            st.session_state.continue_search = True

# --- Main content area: Sorting Visualizer ---
if 'run_sort' in st.session_state and st.session_state.run_sort:
    st.subheader(f"Visualizing {st.session_state.sort_algo}")
    arr = st.session_state.sort_data
    
    sort_visualizers = {
        "Bubble Sort": bubble_sort_visualizer(arr),
        "Selection Sort": selection_sort_visualizer(arr),
        "Insertion Sort": insertion_sort_visualizer(arr),
        "Merge Sort": merge_sort_visualizer(arr, 0, len(arr) - 1),
        "Quick Sort": quick_sort_visualizer(arr, 0, len(arr) - 1),
        "Heap Sort": heap_sort_visualizer(arr),
        "Counting Sort": counting_sort_visualizer(arr),
        "Radix Sort": radix_sort_visualizer(arr),
        "Bucket Sort": bucket_sort_visualizer(arr),
        "Shell Sort": shell_sort_visualizer(arr),
    }

    if st.session_state.sort_algo not in sort_visualizers:
        st.error("Selected sorting algorithm is not implemented.")
    else:
        # Check if a generator is already in session state for continuation
        if 'sort_generator' not in st.session_state or st.session_state.continue_sort:
            if 'sort_generator' not in st.session_state:
                st.session_state.sort_generator = sort_visualizers[st.session_state.sort_algo]
            
            step_placeholder = st.empty()
            array_placeholder = st.empty()
            message_log_placeholder = st.empty()

            for step_info in st.session_state.sort_generator:
                if not st.session_state.continue_sort:
                    st.stop()
                
                current_arr, i_val, idx1, idx2, message, is_swap = step_info
                
                st.session_state.current_message_log.append(message)
                
                display_array_html = []
                for k, val in enumerate(current_arr):
                    css_class = ""
                    
                    # Highlighting logic for sorting algorithms
                    if 'Sort Complete' in message:
                        css_class = "sorted-element"
                    
                    if st.session_state.sort_algo == "Bubble Sort":
                        if i_val != -1 and k >= len(current_arr) - 1 - i_val:
                            css_class = "sorted-element"
                        if k == idx1 or k == idx2:
                            css_class = "swap-element" if is_swap else "compare-element"

                    elif st.session_state.sort_algo == "Selection Sort":
                        if i_val != -1 and k < i_val:
                            css_class = "sorted-element"
                        if k == idx1 or k == idx2:
                            css_class = "swap-element" if is_swap else "compare-element"

                    elif st.session_state.sort_algo == "Insertion Sort":
                        if i_val != -1 and k < i_val:
                            css_class = "sorted-element"
                        if k == idx1 or k == idx2:
                            css_class = "compare-element" if not is_swap else "swap-element"
                    
                    elif st.session_state.sort_algo == "Quick Sort":
                        low, high, pivot_idx, is_swap = step_info[1], step_info[2], step_info[3], step_info[5]
                        if low <= k <= high:
                            css_class = "active-element"
                        if k == pivot_idx:
                            css_class = "compare-element"
                        if is_swap and (k == idx1 or k == idx2):
                            css_class = "swap-element"

                    elif st.session_state.sort_algo == "Merge Sort":
                        low, high, mid = step_info[1], step_info[2], step_info[3]
                        if low != -1 and low <= k <= high:
                            css_class = "active-element"
                    
                    elif st.session_state.sort_algo == "Heap Sort":
                        if k == idx1 or k == idx2:
                            css_class = "swap-element" if is_swap else "compare-element"
                    
                    elif st.session_state.sort_algo == "Counting Sort":
                        if k == idx1 or k == idx2:
                            css_class = "compare-element"

                    elif st.session_state.sort_algo == "Radix Sort":
                        if 'Phase' in message:
                            css_class = "sorted-element"

                    elif st.session_state.sort_algo == "Bucket Sort":
                        if 'Distribution' in message and k == idx1:
                            css_class = "compare-element"
                        if 'Gathering' in message and k == idx1:
                            css_class = "sorted-element"

                    elif st.session_state.sort_algo == "Shell Sort":
                        if k == idx1 or k == idx2:
                            css_class = "swap-element" if is_swap else "compare-element"

                    display_array_html.append(f"<span class='array-element {css_class}'>{val}</span>")
                
                message_log_placeholder.markdown("<br>".join(st.session_state.current_message_log), unsafe_allow_html=True)
                array_placeholder.markdown(f"<div class='array-container'>{' '.join(display_array_html)}</div>", unsafe_allow_html=True)
                time.sleep(0.7)

            st.success("Sorting Complete!")
            st.write(f"Final Sorted Array: {arr}")
            st.session_state.run_sort = False
            del st.session_state.sort_generator
                
# --- Main content area: Searching Visualizer ---
if 'run_search' in st.session_state and st.session_state.run_search:
    st.subheader(f"Visualizing {st.session_state.search_algo}")
    data_list = st.session_state.search_data
    target = st.session_state.search_target
    
    if st.session_state.search_algo != "Linear Search":
        sorted_data = sorted(list(set(data_list)))
        st.info("The data has been sorted to run this algorithm.")
    else:
        sorted_data = data_list
    
    search_visualizers = {
        "Linear Search": linear_search_visualizer(sorted_data, target),
        "Binary Search": binary_search_visualizer(sorted_data, target),
        "Jump Search": jump_search_visualizer(sorted_data, target),
        "Interpolation Search": interpolation_search_visualizer(sorted_data, target),
        "Exponential Search": exponential_search_visualizer(sorted_data, target),
        "Ternary Search": ternary_search_visualizer(sorted_data, target),
    }

    if st.session_state.search_algo not in search_visualizers:
        st.error("Selected search algorithm is not implemented.")
    else:
        # Define placeholders outside the loop
        step_placeholder = st.empty()
        array_placeholder = st.empty()
        message_log_placeholder = st.empty()
        
        found_index = -1
        
        # Display the initial state before starting the loop
        if 'search_generator' not in st.session_state:
            st.session_state.search_generator = search_visualizers[st.session_state.search_algo]
            initial_step = next(st.session_state.search_generator)
            message = initial_step[-1]
            st.session_state.current_message_log.append(message)
            
            # Render initial state
            display_array_html = []
            for k, val in enumerate(sorted_data):
                css_class = ""
                if st.session_state.search_algo == "Binary Search":
                    if 'low' in message:
                        low, high = initial_step[0], initial_step[1]
                        if low <= k <= high:
                            css_class = "active-element"
                elif st.session_state.search_algo == "Linear Search":
                    pass # Linear search doesn't have an initial "active" range
                
                display_array_html.append(f"<span class='array-element {css_class}'>{val}</span>")
            
            message_log_placeholder.markdown("<br>".join(st.session_state.current_message_log), unsafe_allow_html=True)
            array_placeholder.markdown(f"<div class='array-container'>{' '.join(display_array_html)}</div>", unsafe_allow_html=True)
            time.sleep(0.7)

        if st.session_state.continue_search:
            for step_info in st.session_state.search_generator:
                if not st.session_state.continue_search:
                    st.stop()
                
                message = step_info[-1]
                st.session_state.current_message_log.append(message)
                display_array_html = []
                
                # Initialize mid and found to default values to prevent NameError
                current_idx, low, high, mid1, mid2, mid, found = -1, -1, -1, -1, -1, -1, False

                if st.session_state.search_algo == "Linear Search":
                    current_idx, found, _ = step_info
                elif st.session_state.search_algo == "Binary Search":
                    low, high, mid, found, _ = step_info
                elif st.session_state.search_algo == "Interpolation Search":
                    low, high, mid, found, _ = step_info
                elif st.session_state.search_algo == "Jump Search":
                    low, high, mid, found, _ = step_info
                elif st.session_state.search_algo == "Exponential Search":
                    low, high, mid, found, _ = step_info[1:]
                elif st.session_state.search_algo == "Ternary Search":
                    low, high, mid1, mid2, found, _ = step_info
                    if found:
                        mid = mid1 if mid1 != -1 and sorted_data[mid1] == target else mid2
                
                # Check if target was found in the current step to update found_index
                if found:
                    if st.session_state.search_algo == "Linear Search":
                        found_index = current_idx
                    elif st.session_state.search_algo == "Ternary Search":
                        found_index = mid if mid != -1 else (mid1 if mid1 != -1 else mid2)
                    else:
                        found_index = mid

                for k, val in enumerate(sorted_data):
                    css_class = ""
                    
                    # Highlighting logic for searching algorithms
                    if st.session_state.search_algo == "Linear Search":
                        if k <= current_idx:
                             css_class = "active-element"
                        if k == current_idx:
                            css_class = "found-element" if found else "compare-element"
                    else:
                        if low <= k <= high and low != -1:
                            css_class = "active-element"
                        if k == mid and mid != -1:
                            css_class = "found-element" if found else "compare-element"
                        
                        if st.session_state.search_algo == "Ternary Search" and k == mid2 and mid2 != -1 and not found:
                            css_class = "compare-element"

                    display_array_html.append(f"<span class='array-element {css_class}'>{val}</span>")
                
                message_log_placeholder.markdown("<br>".join(st.session_state.current_message_log), unsafe_allow_html=True)
                array_placeholder.markdown(f"<div class='array-container'>{' '.join(display_array_html)}</div>", unsafe_allow_html=True)
                time.sleep(0.7)
                
                if found:
                    break
            
            # Finalize the display and state after the loop completes
            if found_index != -1:
                st.success(f"Target **{target}** found at index **{found_index}**!")
            else:
                st.error(f"Target **{target}** not found.")
            st.session_state.run_search = False
            if 'search_generator' in st.session_state:
                del st.session_state.search_generator
        else:
            # When stopped, display the current state from session_state
            st.warning("Visualization paused. Click 'Continue Search' to resume.")
            
            if 'search_data' in st.session_state and 'search_target' in st.session_state:
                sorted_data = sorted(list(set(st.session_state.search_data))) if st.session_state.search_algo != "Linear Search" else st.session_state.search_data
                
                current_array = []
                last_step = st.session_state.current_message_log[-1] if st.session_state.current_message_log else ""
                
                # Re-apply highlighting based on the last message
                display_array_html = []
                for k, val in enumerate(sorted_data):
                    css_class = ""
                    
                    # This logic is a simplified version for a "stopped" state
                    # For a truly accurate pause, the generator state would need more info
                    if 'found' in last_step.lower():
                        css_class = "found-element"
                    elif 'comparing' in last_step.lower() or 'probe' in last_step.lower() or 'checking' in last_step.lower():
                        css_class = "compare-element"

                    display_array_html.append(f"<span class='array-element {css_class}'>{val}</span>")
                
                # Render the last state
                message_log_placeholder.markdown("<br>".join(st.session_state.current_message_log), unsafe_allow_html=True)
                array_placeholder.markdown(f"<div class='array-container'>{' '.join(display_array_html)}</div>", unsafe_allow_html=True)
                st.stop()

st.markdown("---")
with st.expander("About the Algorithms"):
    st.info("""
    ### Sorting Algorithms:
    - **Bubble Sort:** Repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. Simple but inefficient.
    - **Selection Sort:** Finds the minimum element from the unsorted part and places it at the beginning.
    - **Insertion Sort:** Builds the sorted array one item at a time by inserting each element into its correct position.
    - **Merge Sort:** A divide-and-conquer algorithm that recursively splits the array, sorts each half, and then merges them.
    - **Quick Sort:** Picks a pivot and partitions the array around it. A highly efficient, in-place sort.
    - **Heap Sort:** Uses a binary heap data structure to build a max-heap and repeatedly extracts the maximum element.
    - **Counting Sort:** An integer sorting algorithm that counts occurrences of each element to place them in the correct output array.
    - **Radix Sort:** Sorts integers by processing digits from least significant to most significant.
    - **Bucket Sort:** Distributes elements into buckets, sorts each bucket, and then concatenates the results.
    - **Shell Sort:** An optimization of Insertion Sort that allows for swaps of far-apart elements.

    ### Searching Algorithms:
    - **Linear Search:** Checks each element sequentially. Simple but slow for large datasets.
    - **Binary Search:** (Requires sorted data) Repeatedly divides the search interval in half. Very efficient.
    - **Jump Search:** (Requires sorted data) Jumps ahead by fixed steps, then performs a linear search in the identified block.
    - **Interpolation Search:** (Requires sorted and uniformly distributed data) Estimates the target's position based on its value relative to the array's endpoints.
    - **Exponential Search:** (Requires sorted data) Finds a range exponentially and then performs a binary search within that range.
    - **Ternary Search:** (Requires sorted data) Divides the array into three parts and eliminates two-thirds of the search space in each step.
    """)
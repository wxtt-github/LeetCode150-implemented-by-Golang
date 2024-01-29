[题单:Leetcode150题](https://leetcode.cn/studyplan/top-interview-150/)

#### 数组/字符串

1. 合并两个有序数组

```go
func merge(nums1 []int, m int, nums2 []int, n int)  {
    /*思路:从后往前,每次比较末尾元素,较大的放在nums1的末尾,
    同时,为了满足题意,当两元素相等时,优先放入nums2的元素
    */
    for i,j,k := m-1,n-1,m+n-1;j >= 0;k--{
        if i >= 0 && nums1[i] > nums2[j]{
            nums1[k] = nums1[i]
            i--
        } else{
            nums1[k] = nums2[j]
            j--
        }
    }
}``
```

2. 移除元素

```go
func removeElement(nums []int, val int) int {
	/*思路:使用首尾双指针,首指针left寻找等于val的值,尾指针right寻找
	  不等于val的值,各找到一个后进行交换,循环方式类似快排
	*/
	left := 0
	right := len(nums) - 1
	for left <= right {
		for left <= right && nums[left] != val {
            left++
		}
        for left <= right && nums[right] == val{
            right--
        }
        if left <= right{
            temp := nums[left]
            nums[left] = nums[right]
            nums[right] = temp
            left++
            right--
        }
	}
    return left
}
```

3. 删除有序数组中的重复项

```go
func removeDuplicates(nums []int) int {
	/*思路:使用快慢双指针,初始时慢指针指向0,快指针指向1,
	  我们认为慢指针及之前的元素都是不重复的,基于这个思想,
	  让快指针寻找到与慢指针不同的元素,然后慢指针加1,赋值即可
	*/
	len := len(nums)
    if len == 1{
        return 1
    }

	slow := 0
	quick := 1
	for quick < len {
		for quick < len && nums[quick] == nums[slow] {
			quick++
		}
		if quick < len {
			slow++
			nums[slow] = nums[quick]
			quick++
		}
	}
	return slow + 1
}
```

4. 删除有序数组中的重复项II

```go
func removeDuplicates(nums []int) int {
    /*思路:同上题,采用快慢双指针思路,慢指针slow及之前的元素为重复但不超过
    2次的元素,而快指针quick用于寻找重复不超过两次的元素,可以将quick与slow,slow-1
    做对比,若相等则跳过,其余同上题
    */
    slow := 1
    quick := 2
    length := len(nums)
    if length == 1{
        return 1
    }
    if length == 2{
        return 2
    }

    for slow <= quick && quick < length{
        for quick < length && nums[quick] == nums[slow] && nums[quick] == nums[slow-1]{
            quick++
        }
        if quick < length{
            slow++
            nums[slow] = nums[quick]
            quick++
        }
    }
    return slow+1
}
```

5. 多数元素

```go
func majorityElement(nums []int) int {
	/*思路:由于数组总是非空的且总存在多数元素,采用摩尔投票法,
	  首先定义票数最多的元素为major为第一个元素,其count为1,
	  遍历之后的数组,若与major元素相同,则count++,若不同,则count--,
	  当count为0时,若不同,则修改major元素,这是基于总存在多数元素,
	  因此可以这么修改得出正确结果
	*/
	major := nums[0]
	cnt := 1
	for i := 1; i < len(nums); i++ {
        if nums[i] == major{
            cnt++
        } else if nums[i] != major && cnt == 0{
            major = nums[i]
            cnt = 1
        } else{
            cnt--
        }
	}
    return major
}
```

#### 双指针

1. 验证回文串

`注:将字符串改为大写的函数为strings.ToUpper(<string>)`

```go
func isPalindrome(s string) bool {
	/*思路:使用首尾双指针,用strings.ToLower(s)先将字符串转为小写,
	  然后判断双指针指向的是否有效,若无效做进行自增自减操作即可
	*/
    s = strings.ToLower(s)
	left := 0
	right := len(s) - 1
	for left < right {
        if !isValid(s[left]){
            left++
            continue
            //continue的目的是重新进入for循环,这样能实现一直++直到出现有效字符的功能
        }
        if !isValid(s[right]){
            right--
            continue
        }
        if s[left] != s[right]{
            return false
        }
        left++
        right--
	}
    return true
}

func isValid(c byte) bool {
	if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
		return true
	}else{
        return false
    }
}
```

2. 判断子序列

```go
func isSubsequence(s string, t string) bool {
    /*思路1:双指针,分别指向子串s和主串t,若能匹配则两指针均往后移,
    若无法匹配则只有主串的指针往后移
    思路2:貌似还有动态规划求解,我觉得双指针比较简单就不写了
    */
    ps := 0
    pt := 0
    for ps < len(s) && pt < len(t){
        if s[ps] == t[pt]{
            ps++
            pt++
        }else{
            pt++
        }
    }
    if ps == len(s){
        return true
    }else{
        return false
    }
}
```

3. 两数之和 II - 输入有序数组

```go
func twoSum(numbers []int, target int) []int {
    /*思路:采用首尾双指针,因为序列是非递减的且题目说明了必有答案,
    因此可计算首尾指针之和sum,若sum小于target,说明sum不够大,因此将首指针后移,
    若sum过小,则将尾指针前移,其中的原理在于,如首指针所指的值过小,则前面的也均不匹配
    */
    left := 0
    right := len(numbers)-1
    for left < right && numbers[left]+numbers[right] != target{
        sum := numbers[left] + numbers[right]
        if sum < target{
            left++
        }else if sum > target{
            right--
        }
    }
    array := []int{left+1,right+1}
    return array
}
```

4. 盛最多水的容器

```go
func maxArea(height []int) int {
    /*思路:使用首尾双指针,首先对于初始状态,宽是最大的,之后宽只会越变越小,
    这时候我们根据双指针的高度,尽可能抛弃较小的高度,直到双指针会合停止
    */
    left := 0
    right := len(height)-1
    var MaxArea int = 0
    for left < right{
        tempArea := (right-left) * min(height[left],height[right])
        MaxArea = max(MaxArea,tempArea)
        if height[left] < height[right]{
            left++
        }else{
            right--
        }
    }
    return MaxArea
}

func min(a,b int) int{
    if a < b{
        return a
    }else{
        return b
    }
}
func max(a,b int) int{
    if a > b{
        return a
    }else{
        return b
    }
}
```

5. 三数之和

```go
func threeSum(nums []int) [][]int {
    /*思路:首先要把三数之和问题转化为两数之和问题,固定一个数a,
    则另外两个数b和c之和等于-a时,三数之和为0,由此转化为两数之和问题。
    首先对数组进行排序，分情况讨论，遍历数组每一个元素，若第一个数大于0，
    则说明不可能存在三数之和为0的数，直接返回。由于题目要求不重复的三元组，
    因此除了第一个数以外，可以检查当前数是否与上一个数相等，若相等则continue，
    若不相等则用双指针解决两数之和问题，注意两数之和同样要去重
    */
    sort.Ints(nums)
    var array [][]int
    if len(nums) < 3{
        return array
    }
    for index,value := range nums{
        if value > 0{
            return array
        }
        if index > 0 && nums[index] == nums[index-1]{
            continue
        }
        left := index+1
        right := len(nums)-1
        for left < right{
            sum := value + nums[left] + nums[right]
            if sum == 0{
                array = append(array,[]int{value,nums[left],nums[right]})
                for left < right && nums[left] == nums[left+1]{
                    left++
                }
                for left < right && nums[right] == nums[right-1]{
                    right--
                }
                left++
                right--
            }else if sum < 0{
                left++
            }else if sum > 0{
                right--
            }
        }
    }
    return array
}
```

#### 滑动窗口

1. 长度最小的子数组

```go
func minSubArrayLen(target int, nums []int) int {
    /*思路：采用滑动窗口的思想，首先定义两个指针start和end，均指向0，
    然后定义sum，先让sum加上end指的内容，若此时sum满足大于等于target，
    则与min长度进行比较，若小于则进行更新，不小于则不管，若sum依然小于target，
    则让end后移，同时加上sum加上end的内容，若sum大于等于target，
    则还要将start往后移，并减去start所指的内容，由此得到结果
    */
    start := 0
    end := 0
    length := len(nums)
    minLength := length+1
    if len(nums) == 0{
        return 0
    }
    sum := 0
    for start < length && end < length{
        sum += nums[end]
        for sum >= target{
            minLength = min(minLength,end-start+1)
            sum-=nums[start]
            start++
        }
        end++
    }
    if minLength == length+1{
        return 0
    }else{
        return minLength
    }
}

func min(x int,y int) int{
    if x < y{
        return x
    }else{
        return y
    }
}
```

2. 无重复字符的最长子串

```go
func lengthOfLongestSubstring(s string) int {
	/*思路：使用滑动窗口的思想，首先定义两个指针i和right，
	  i初始指向0，right初始指向-1，定义一个哈希集合，
	  m := map[byte]int{}，用于记录每个字符出现次数，
	  将i从0遍历到s的长度，若i不为0，首先进行delete(m,s[i-1])操作，
	  （因为在for循环里i++），然后不断右移right指针，直到找到不重复出现的字符，
	  此时不重复子串的长度为right-i+1，再将其与原先的max长度相比较即可
      
      时间复杂度：O(n)
      空间复杂度：O(ASCLL码的范围大小)
	*/
	m := map[byte]int{}
	length := len(s)
	maxLength := 0
	right := -1
	for i := 0; i < length; i++ {
        if i != 0{
            delete(m,s[i-1])
        }
        for right+1 < length && m[s[right+1]]==0{
            m[s[right+1]]++
            right++
        }
        maxLength = max(maxLength,right-i+1)
	}
    return maxLength
}

func max(x,y int) int {
    if x < y{
        return y
    }else{
        return x
    }
}
```

3. 串联所有单词的子串(2ms，得意的一题)

```go
func findSubstring(s string, words []string) []int {
	/*思路：这是一道难度很高的题，采用滑动窗口的思想，首先定义一个map，
	  存放words中各个单词出现的次数。由于words中各个元素长度k相等，因此
	  我们将s进行划分处理，每次对单个元素的长度进行增加，一个个单词地处理。
	  首先定义一个for循环，i取0，i小于单个元素长度k，i自增，因为后续还有一层
	  for循环，以k自增，因此为了避免重复。i从0到k-1循环即可。
	  定义计数变量cnt初值为0，定义一个map，名叫counterMap，
	  定义一个左指针left用于记录初始位置方便后续append进数组里，
	  定义一个右指针right用于进行取单词处理，两者初值均为i，
	  right满足right<=len(s)-k，每次自增k，取单词word:=s[right:right+k]，
	  用一个if num,found := m[word];found{}，num代表word的值，found代表是否
	  查找成功，因此当查找成功时，我们首先判断counterMap中word的值是否大于等于num，
	  若大于等于，则说明单词数超标了，将counterMap中s[l:l+k]减1，cnt--，l+=k，
	  这步是为了防止比如words有"a"和"b"，这时候需要的是一个a和一个b，不是两个a。
	  然后将counterMap[word]++，count++。
	  当查找失败时，说明当前单词不在map里，将left指针右移至right+k的位置，同时清除
	  counterMap中的元素，cnt--，最后由于是for循环，right会+=k,因此又会从left==right
	  的情况开始。
	  做完上面两个判断后，判断cnt是否等于words的长度，若等于则将left指针append进数组中
	*/
	arr := []int{}
	lenOfS := len(s)
	lenOfWords := len(words)
	if lenOfWords == 0 {
		return arr
	}
	lenOfSingle := len(words[0])
	if lenOfS < lenOfWords*lenOfSingle {
		return arr
	}

	m := map[string]int{}
	for _, temp := range words {
		m[temp]++
	}
	for i := 0; i < lenOfSingle; i++ {
        cnt := 0
        counterMap := map[string]int{}
        for left,right := i,i;right <= lenOfS-lenOfSingle;right+=lenOfSingle{
            tempWord := s[right:right+lenOfSingle]
            if num,found := m[tempWord];found{
                for counterMap[tempWord] >= num{
                    counterMap[s[left:left+lenOfSingle]]--
                    cnt--
                    left+=lenOfSingle
                }
                counterMap[tempWord]++
                cnt++
            }else{
                for left < right{
                    counterMap[s[left:left+lenOfSingle]]--
                    cnt--
                    left+=lenOfSingle
                }
                left+=lenOfSingle
            }
            if cnt == lenOfWords{
                arr = append(arr,left)
            }
        }
	}
    return arr
}
```

4. 最小覆盖子串

```go
func minWindow(s string, t string) string {
	/*思路：滑动窗口思想，定义两指针left和right，包裹起来的为滑动窗口，
	  主体思路是先使right右移，直到找到一个可行解，然后再让left右移，
	  若右移left依然能够包括t，则继续右移，直到无法包括，此时找到了一个局部最优解，
	  找到局部最优解后，用该局部最优解去尝试更新最优解，然后将左指针右移一位，使得该
	  窗口刚好不能包含t中所有字符，开始下一轮寻找。
	  具体实现是首先定义一个map，map中byte的值负数代表相比较t缺了多少个，0表示正好，
	  正数表示相比较t多了多少个。首先对t遍历，将map中的值进行自减，定义diff作为map中
	  byte的种类。
	  然后定义一个for循环，left和right初值为0，right<s的长度，每次right++，
	  设置if对s[right]在map中查找，查找成功则将其值++，若正好++后为0，则diff--。
	  其中再写一个for，条件是diff==0，每次left++，设置if对s[left]在map中查找，
	  若查找成功则将其值--，若正好减为-1，则将diff++，此时找到局部最优解，
	  若长度小于最短长度，则将字符串改为s[left:right+1]，长度更新为right-left+1。
      时间复杂度：O(n)，n为s长度
      空间复杂度：O(m)，m为t中字符种类
	*/
	str := ""
	minLength := len(s) + 1
	m := make(map[byte]int)
	lenOfT := len(t)
	for i := 0; i < lenOfT; i++ {
		m[t[i]]--
	}
	diff := len(m)
	for left, right := 0, 0; right < len(s); right++ {
        if _,found := m[s[right]];found{
            m[s[right]]++
            if m[s[right]] == 0{
                diff--
            }
        }
        for diff == 0{
            if _,found := m[s[left]];found{
                m[s[left]]--
                if m[s[left]] == -1{
                    diff++
                    if right-left+1 < minLength{
                        minLength = right-left+1
                        str = s[left:right+1]
                    }
                }
            }
            left++
        }
	}
    return str
}
```

#### 矩阵

1. 有效的数独

```go
func isValidSudoku(board [][]byte) bool {
	/*思路：对于一个i行j列的数组元素，如果他不是'.'，采用计数的思想，
	  创建两个9*9的二维矩阵row和column，第一个9用于表示多少行或列，
	  第二个9用于表示数字几出现的次数，如row[0][0]=2代表第1行的1元素
	  出现了两次，因此不符题意，返回true。
	  本题难点在每一个3*3的小九宫格sub，创建一个3*3*9的三维矩阵，
	  前两个3表示第几行第几列的小九宫格（把每个小九宫格看做整体），
	  后面的9代表数字几出现的次数，如sub[0][0][1]=0代表第一行第一列的
	  小九宫格中1出现了0次。
	  因此对于i行j列的元素m，我们只需要将row[i][m-1]++，column[j][m-1]++，
	  sub[i/3][j/3][m-1]++即可，若出现其中某个大于1时，直接返回false
	*/
	rows := [9][9]int{}
    //还有一种等价写法是var rows [9][9]int，其他的make，或者省略{}都是不合法的
	columns := [9][9]int{}
	subs := [3][3][9]int{}
	for i, row := range board {
		for j, val := range row {
			if val == '.' {
				continue
			}
			index := val - '1'
			rows[i][index]++
			columns[j][index]++
			subs[i/3][j/3][index]++
			if rows[i][index] > 1 || columns[j][index] > 1 || subs[i/3][j/3][index] > 1 {
                return false
			}
		}
	}
    return true
}
```

2. 螺旋矩阵

```go
func spiralOrder(matrix [][]int) []int {
	/*思路：模拟螺旋过程，定义left，right，top，bottom，初值分别为
		0，len(matrix[0])-1,0,len(matrix)-1，其中len(matrix)代表matrix有几行，
		然后遵循从左往右，从上往下，从右往左，从下往上的顺序，最外层for循环终止条件
		是遍历完(right+1)*(bottom+1)个元素，用number来控制遍历完这些元素，防止继续
	    进行for循环
        时间复杂度：O(矩阵元素数)
        空间复杂度：不算arr的话O(1)，算的话O(矩阵元素数)
	*/
	arr := []int{}
	left, right, top, bottom := 0, len(matrix[0])-1, 0, len(matrix)-1
	number := (right + 1) * (bottom + 1)
	i := 0
	for i < number {
		for j := left; j <= right && i < number; j++ {
			arr = append(arr, matrix[top][j])
			i++
		}
		top++
		//从左往右，贴着上墙走
		for j := top; j <= bottom && i < number; j++ {
			arr = append(arr, matrix[j][right])
			i++
		}
		right--
		//从上往下，贴着右墙走
		for j := right; j >= left && i < number; j-- {
			arr = append(arr, matrix[bottom][j])
			i++
		}
		bottom--
		//从右往左，贴着下墙走
		for j := bottom; j >= top && i < number; j-- {
			arr = append(arr, matrix[j][left])
			i++
		}
		left++
		//从下往上，贴着左墙走
	}
	return arr
}
```

3. 旋转图像

```go
func rotate(matrix [][]int)  {
    /*思路：顺时针旋转90度相当于先垂直翻转（以水平线为轴上下翻转），
    再对角线翻转。同理旋转180,270只需要多旋转几次即可
    */
    n := len(matrix)
    for i := 0;i < n/2;i++{
        for j := 0;j < n;j++{
            matrix[i][j],matrix[n-1-i][j] = matrix[n-1-i][j],matrix[i][j]
        } 
    }
    //垂直翻转
    for i := 0;i < n;i++{
        for j := i;j < n;j++{
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        }
    }
    //对角线翻转
}
```

4. 矩阵置零

```go
func setZeroes(matrix [][]int) {
	/*思路：对于一个矩阵，我们先不看第0行和第0列，
	  从第1行第1列以后的子矩阵往后看，对其遍历元素i，j，若找到0，则可以标记
	  其i行0列，0行j列的元素为0，之后我们重新遍历元素i，j，若发现其i行0列或
	  0行j列的元素为0，则将其置为0。这样我们可以利用第0行和0列的空间来储存信息。
	  但是这样我们处理了1行1列后的子矩阵，第0行和第0列没解决，因此我们一开始就
	  遍历第0行和第0列，设置两个变量用来记录是否存在0，再根据这个变量来处理0行0列
	  时间复杂度：O(mn)
	  空间复杂度：O(1)
	*/
	rows := len(matrix)
	columns := len(matrix[0])
	isRowExist0 := false
	isColumnExist0 := false

	for i := 0; i < columns; i++ {
		if matrix[0][i] == 0 {
			isRowExist0 = true
			break
		}
	}
	for i := 0; i < rows; i++ {
		if matrix[i][0] == 0 {
			isColumnExist0 = true
			break
		}
	}
	//判断第0行0列是否有0
	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j] = 0
				matrix[i][0] = 0
			}
		}
	}
	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	//处理1行1列后的子矩阵
	if isRowExist0 {
		for i := 0; i < columns; i++ {
			matrix[0][i] = 0
		}
	}
	if isColumnExist0 {
		for i := 0; i < rows; i++ {
			matrix[i][0] = 0
		}
	}
}
```

5. 生命游戏

```go

```

#### 哈希表

1. 赎金信

```go

```

2. 同构字符串

```go

```

3. 单词规律

```go

```

4. 有效的字母异位词

```go

```

5. 字母异位词分组

```go

```

6. 两数之和

```go

```

7. 快乐数

```go

```

8. 存在重复元素II

```go

```

9. 最长连续序列

```go

```

#### 区间

1. 汇总区间

```go

```

2. 合并区间

```go

```

3. 插入区间

```go

```

4. 用最少数量的箭引爆气球

```go

```


using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class UIText : MonoBehaviour
{
        public static UIText instance;

    public Text CanvasText;

    public void Awake()
    {
        instance = this;
    }

    void Start()
    {
        CanvasText.text = "_";
    }

    public void Calibrating()
    {
        CanvasText.text = "Calibrating.......";
    }

    public void Left()
    {
        CanvasText.text = "Left Hand Selected";
    }

    public void Right()
    {
        CanvasText.text = "Right Hand Selected";
    }

}

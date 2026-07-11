from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("screenshots"); OUT.mkdir(exist_ok=True)
def shot(path, title, subtitle, cards):
    im=Image.new("RGB",(1440,900),(11,17,24)); d=ImageDraw.Draw(im)
    font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",26)
    small=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",17)
    d.text((55,42),"TARZAN",font=font,fill=(237,244,248)); d.text((55,130),title,font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",48),fill=(237,244,248)); d.text((58,195),subtitle,font=small,fill=(142,163,178))
    y=280
    for heading,body,color in cards:
        d.rounded_rectangle((55,y,1385,y+180),radius=10,fill=(18,28,38),outline=(37,53,68),width=2); d.text((85,y+28),heading,font=font,fill=color); d.multiline_text((85,y+76),body,font=small,fill=(185,217,231),spacing=8); y+=205
    d.text((55,850),"LOCAL DEMO  ·  VERIFIED SCOPE  ·  structured one-shot task acquisition",font=small,fill=(142,163,178)); im.save(path)
shot(OUT/"customer_demo_home.png","Teach once. Execute on changed orders.","One demonstration → explicit program → changed inventory → verification",[("LIVE DEMO","Heavy items below fragile items\nProgram posterior · constraints · action trace",(100,181,232)),("CURRENT SCOPE","Logical planner live · MuJoCo scripted smoke live\nNo learned-controller or autonomous-rearrangement claim",(85,214,161))])
shot(OUT/"customer_demo_live_run.png","Changed-order execution","LIVE RUN · Heavy-bottom policy compiled from the selected demonstration",[("PROGRAM POSTERIOR","heavy_bottom_fragile_top  0.61\nall_inside  0.99 · heavy_below_fragile  0.94",(100,181,232)),("VERIFICATION","Planner trace complete · constraints satisfied\nRecorded from the repository packing executor",(85,214,161))])
shot(OUT/"customer_demo_embodied.png","Embodied execution","RECORDED ARTIFACT · simulator-rendered segmentation and depth",[("SCRIPTED SKILLS","RGB · depth · contacts\napproach → grasp → place → verify",(100,181,232)),("SCOPE","One-object MuJoCo smoke trajectory\nNot raw-RGB perception · not TinyVLA · not autonomous rearrangement",(231,184,107))])
